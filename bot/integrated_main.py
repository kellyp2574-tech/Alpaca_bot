"""Combined Overnight Rebound Bot — Main Orchestrator

3-Sleeve Optimized Portfolio:
  Sleeve 1 (Intraday ETF)  — one trade per day at 10:00 or 10:10, flat by 15:30
  Sleeve 2 (Overnight ETF) — entered at 15:45, exited at 09:30, only if no intraday trade
  Sleeve 3 (Single-stock MR) — entered at 15:45, only if no intraday OR overnight ETF trade

Daily Schedule (ET) — bot starts 09:00 AM:

MORNING (T+1 exits — positions from yesterday's 15:45 entries):
  09:00  Start, detect overnight positions from broker
  09:25  Cancel any resting open orders, freeze broker exit plan
  09:30  Submit batched market sells for all remaining overnight positions
  09:31  Broker-native rescue pass for any remaining positions
  09:45  Post-exit failsafe — verify broker is flat or force-flatten stragglers

INTRADAY ETF (one sleeve per day):
  09:30-10:00  Build ETF tape (QQQ, SPY, VXX, SVIX, TQQQ)
  10:00  Evaluate strategies 1-3 (VXX Spike Recovery, VXX Collapse, Momentum)
  10:00  Enter immediately if strategy 1-3 fires
  10:10  Evaluate strategies 4-5 (Router Long, SVIX Long) if no trade yet
  10:10  Enter immediately if strategy 4-5 fires
  15:00  Exit TQQQ/SVIX for strategies 3/4/5
  15:30  Exit TQQQ for strategies 1/2; hard flatten any remaining intraday ETF

AFTERNOON — only if NO intraday ETF trade was taken:
  15:45  Evaluate overnight ETF strategies A/B/C (VXX MR→SVIX, Quality→TQQQ, Gap Bounce→TQQQ)
  15:45  If overnight ETF fires, enter ETF position (blocks single-stock MR)
  15:45  If overnight ETF does NOT fire, run single-stock MR entry pipeline
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
from bot import overnight_etf_runner
from bot import scoring
from bot import state_io
from bot.massive_client import MassiveClient
from bot.market_data import AlpacaDataClient
from bot.mean_reversion_scorer import MeanReversionCandidate
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
    (9, 24, 9, 32),   # Order cancel, 09:30 batch sells, 09:31 rescue
    (9, 58, 10, 12),  # ETF tape final, 10:00 strat 1-3, 10:10 strat 4-5
    (14, 58, 15, 2),  # 15:00 intraday exit (strats 3/4/5)
    (15, 28, 15, 32), # 15:30 intraday hard flatten (strats 1/2)
    (15, 44, 16, 1),  # Overnight ETF + MR selection, entry, EOD
)


def _is_hot_window(now_t: dt_time) -> bool:
    """True if ``now_t`` falls inside any pre-defined hot window."""
    cur = now_t.hour * 60 + now_t.minute
    for sh, sm, eh, em in _HOT_WINDOWS_HHMM:
        if (sh * 60 + sm) <= cur < (eh * 60 + em):
            return True
    return False


class CombinedOvernightReboundBot:
    """Unified Bot Orchestrator — Intraday ETF + Overnight ETF + Single-stock MR

    Priority precedence:
      1. Intraday ETF (10:00 strategies 1-3, 10:10 strategies 4-5)
         → any fill blocks both overnight ETF and single-stock MR
      2. Overnight ETF (15:45 strategies A/B/C)
         → any fill blocks single-stock MR
      3. Single-stock MR (15:45)
         → runs only if neither intraday nor overnight ETF fired
    """

    def __init__(self):
        self.massive = MassiveClient()
        self.alpaca = AlpacaDataClient()
        self.position_mgr = PositionManager()
        self.state_mgr = StateManager()
        self.etf_router = ETFRouter(config)

        # Universe & candidates
        self.universe: List[str] = []
        self.mr_candidates: List[MeanReversionCandidate] = []
        self._universe_diag: Optional[UniverseDiagnostics] = None

        # ETF Router state
        self.router_decision: Optional[RouterDecision] = None
        self.router_traded_today = False
        self.router_branch: Optional[str] = None
        self.mr_blocked_today = False
        self.etf_positions: Dict[str, Any] = {}  # keyed by branch value, e.g. {"MOMENTUM_SLEEVE": {...}}
        self.etf_opens_930: Dict[str, float] = {}
        self.tape_recording_active = False
        self._tape_last_update_monotonic: float = 0.0

        # Intraday sleeve
        self.intraday_etf_sleeve_filled = False
        self.router_decision_1010_made = False  # 10:10 check done flag

        # Overnight ETF sleeve (precedence over single-stock MR)
        self.overnight_etf_fired = False        # True if an overnight ETF strategy fired
        self.overnight_etf_position: Optional[Dict[str, Any]] = None
        self.overnight_etf_decision_made = False

        # Stage flags
        self.startup_done = False
        self.tape_initialized = False
        self.router_decision_made = False
        self.morning_exits_done = False
        self.data_collected = False
        self.scoring_done = False
        self.entries_done = False

        # Open-exit state
        self.open_exit_plan: List[Dict[str, Any]] = []  # Frozen broker-position sell plan built after 09:25 cleanup
        self.open_exit_submitted = False  # True once 09:30 batch sell submitted; persisted for restart safety

        # Morning order management
        self.morning_open_orders_cancelled = False
        
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
        # Live MR sleeve: 3 positions x 30% per position = 90% total max.
        per_pos = float(getattr(config, "MR_ALLOC_PER_POSITION_PCT", 0.0))
        total = float(getattr(config, "MR_MAX_TOTAL_ALLOCATION_PCT", 0.0))
        max_pos = int(getattr(config, "MR_MAX_PRIMARY_POSITIONS", 0))
        if not (0.0 < per_pos <= 1.0):
            raise ValueError(f"MR_ALLOC_PER_POSITION_PCT must be in (0, 1], got {per_pos}")
        if not (0.0 < total <= 1.0):
            raise ValueError(f"MR_MAX_TOTAL_ALLOCATION_PCT must be in (0, 1], got {total}")
        if max_pos < 1:
            raise ValueError(f"MR_MAX_PRIMARY_POSITIONS must be >= 1, got {max_pos}")
        if per_pos * max_pos < total - 1e-6:
            logger.warning(
                f"MR sizing: per_pos*{max_pos}={per_pos*max_pos:.2%} < total cap {total:.0%}; "
                f"sleeve cannot reach the configured total cap with this many positions"
            )

        intraday = float(getattr(config, "INTRADAY_ETF_ALLOCATION_PCT", 0.0))
        if not (0.0 < intraday <= 1.0):
            raise ValueError(f"INTRADAY_ETF_ALLOCATION_PCT must be in (0, 1], got {intraday}")

        if _parse_config_time(config.SCORING_TIME) > _parse_config_time(config.ENTRY_TIME):
            raise ValueError(
                f"SCORING_TIME must be <= ENTRY_TIME, got "
                f"{config.SCORING_TIME} > {config.ENTRY_TIME}"
            )

        # IEX volume multiplier warning
        data_feed = str(getattr(config, "DATA_FEED", "")).lower()
        adv_multiplier = float(getattr(config, "ADV_DOLLAR_MULTIPLIER", 1.0) or 1.0)
        min_adv = float(getattr(config, "MR_MIN_AVG_DOLLAR_VOLUME", 0) or 0)

        if data_feed == "iex" and adv_multiplier <= 1.0 and min_adv > 0:
            logger.warning(
                "DATA_FEED=iex but ADV_DOLLAR_MULTIPLIER <= 1.0. "
                "MR liquidity filters may reject valid candidates."
            )

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
        t_exit_all     = _parse_config_time(config.MORNING_EXIT_TIME)       # 09:30 batched market sells
        t_failsafe     = _parse_config_time(config.MORNING_FAILSAFE_TIME)    # 09:45 post-exit failsafe
        t_data_collect = _parse_config_time(config.DATA_COLLECTION_TIME)    # 15:30
        t_scoring      = _parse_config_time(config.SCORING_TIME)            # 15:45
        t_entry        = _parse_config_time(config.ENTRY_TIME)              # 15:45
        t_cancel_orders = _parse_config_time(getattr(config, "MORNING_CANCEL_OPEN_ORDERS_TIME", "09:25"))
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

        if current_time >= t_market_close:
            logger.info("Started after market close — nothing to do until next market day")
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
                # 09:25 — cancel any resting open orders before normal exits.
                # Run regardless of local position state for safety.
                if (not self.morning_open_orders_cancelled
                        and current_time >= t_cancel_orders):
                    logger.warning("09:25 order cleanup: canceling all open orders before 09:30 exits")
                    self.position_mgr.cancel_all_open_orders()
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
                    if current_time >= t_exit_all and not self.open_exit_submitted:
                        self._submit_open_exit_market_sells()
                        self.open_exit_submitted = True
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

                    # 09:45 post-exit failsafe — force-flatten if broker still holds anything.
                    if (not self.post_exit_failsafe_done
                            and current_time >= t_failsafe):
                        bc = self.position_mgr.broker_position_count()
                        if bc > 0:
                            logger.warning(f"Post-exit failsafe: broker still has {bc} positions")
                            self._run_failsafe_flatten(f"{t_failsafe.strftime('%H:%M')} post-exit failsafe")
                        elif bc == 0:
                            logger.info("Post-exit failsafe: broker confirmed flat")
                            self.position_mgr.reconcile_local_positions_from_broker()
                        self.post_exit_failsafe_done = True
                        self.morning_exits_done = True
                        self._save_state()

                    # Early completion: after 09:30 submit, when broker confirmed flat.
                    if current_time >= t_exit_all and not self.morning_exits_done:
                        bc = self.position_mgr.broker_position_count()
                        if bc == 0:
                            logger.info("All exits complete — broker confirmed flat")
                            self.position_mgr.positions.clear()
                            self.morning_exits_done = True
                            self._save_state()

            # ════════════════════════════════════════════
            # INTRADAY ETF Sleeve (5 strategies, one per day)
            # 90% of equity; any fill blocks overnight ETF + MR
            # ════════════════════════════════════════════

            if getattr(config, "ETF_ROUTER_ENABLED", False):
                t_startup       = _parse_config_time(getattr(config, "BOT_START_TIME", "09:00"))
                t_market_open   = _parse_config_time(getattr(config, "MARKET_OPEN_TIME", "09:30"))
                t_router_dec    = _parse_config_time(getattr(config, "ROUTER_DECISION_TIME", "10:00"))
                t_router_1010   = _parse_config_time(getattr(config, "ROUTER_1010_TIME", "10:10"))
                t_intraday_1500 = _parse_config_time(getattr(config, "INTRADAY_EXIT_1500", "15:00"))
                t_intraday_flat = _parse_config_time(getattr(config, "INTRADAY_ETF_HARD_FLATTEN_TIME", "15:30"))

                # 09:00 startup
                if (not self.startup_done
                        and current_time >= t_startup
                        and current_time < t_market_open):
                    self._run_startup_phase()

                # Late-start guard: bot started after market open without
                # completing the 09:00 startup phase — cleanly disable the
                # ETF router so it never enters a half-initialized state.
                if (not self.startup_done
                        and current_time >= t_market_open
                        and not self.router_decision_made):
                    logger.warning(
                        f"Late start at {current_time.strftime('%H:%M')} — "
                        f"startup phase not completed; disabling intraday ETF router for today"
                    )
                    self.startup_done = True
                    self.router_decision_made = True
                    self.router_decision_1010_made = True
                    self.router_traded_today = False
                    self.mr_blocked_today = False
                    self.router_branch = "Late start - router disabled"
                    self._save_state()

                # 09:30 initialize ETF tape
                if (self.startup_done
                        and not self.tape_initialized
                        and not self.router_decision_made
                        and current_time >= t_market_open
                        and current_time < t_router_dec):
                    if current_time > dt_time(9, 31):
                        logger.warning(f"ETF router late-start at {current_time.strftime('%H:%M')} without 09:30 tape; disabling router for today")
                        self.router_decision_made = True
                        self.router_decision_1010_made = True
                        self.router_traded_today = False
                        self.mr_blocked_today = False
                        self.router_branch = "Late start - router disabled"
                        self._save_state()
                    else:
                        self._initialize_tape_recording()

                # 09:30-10:00 tape updates
                if (self.tape_initialized
                        and not self.router_decision_made
                        and current_time >= t_market_open
                        and current_time < t_router_dec):
                    self._update_tape()

                # 10:00 — strategies 1-3 (enter immediately if fired)
                if (self.tape_initialized
                        and not self.router_decision_made
                        and current_time >= t_router_dec):
                    self._update_tape(force=True)
                    self._make_router_decision()

                # 10:00-10:10 continue tape for strategies 4-5
                if (self.router_decision_made
                        and not self.router_decision_1010_made
                        and not self.intraday_etf_sleeve_filled
                        and current_time >= t_router_dec
                        and current_time < t_router_1010):
                    self._update_tape()

                # 10:10 — strategies 4-5 (only if no trade yet)
                if (self.router_decision_made
                        and not self.router_decision_1010_made
                        and not self.intraday_etf_sleeve_filled
                        and current_time >= t_router_1010):
                    self._update_tape(force=True)
                    self._make_router_decision_1010()
                    self.router_decision_1010_made = True

                # ETF exit checkpoints (planned_exit_time per position)
                if self.etf_positions:
                    self._check_etf_exits(current_time)

                # 15:00 intraday hard exit guard (belt-and-suspenders for strats 3/4/5)
                if self.etf_positions and current_time >= t_intraday_1500:
                    for _bk in list(self.etf_positions.keys()):
                        if _bk in ("MOMENTUM_SLEEVE", "MOMENTUM_SLEEVE_ANTI", "ROUTER_LONG", "SVIX_LONG"):
                            logger.warning(f"15:00 hard exit for {_bk}")
                            self._execute_etf_exit(_bk)

                # 15:30 intraday hard flatten — all ETF must be flat before MR entries
                if self.etf_positions and current_time >= t_intraday_flat:
                    for _bk in list(self.etf_positions.keys()):
                        logger.critical(f"Intraday hard flatten at 15:30 — forcing ETF exit ({_bk})")
                        self._execute_etf_exit(_bk)

            # ════════════════════════════════════════════
            # AFTERNOON: Score universe and enter new positions
            # ════════════════════════════════════════════

            # 15:30 — Data collection (universe + bars)
            if not self.data_collected and current_time >= t_data_collect:
                if current_time < dt_time(15, 50):
                    self._step_collect_data()
                else:
                    logger.warning("Past 3:50 PM without data collection — attempting now")
                    self._step_collect_data()

            # 15:45 — Score and rank MR candidates
            if self.data_collected and not self.scoring_done and current_time >= t_scoring:
                self._step_score_and_rank()

            # 15:45 — Overnight ETF (strategies A/B/C) — only if no intraday trade today
            if (self.scoring_done
                    and not self.overnight_etf_decision_made
                    and not self.intraday_etf_sleeve_filled
                    and current_time >= t_entry
                    and getattr(config, "OVERNIGHT_ETF_ENABLED", True)):
                self._evaluate_overnight_etf_strategies()
                self.overnight_etf_decision_made = True
                self._save_state()

            # 15:45 — Single-stock MR — only if neither intraday nor overnight ETF fired
            if (self.scoring_done
                    and not self.entries_done
                    and not self.mr_blocked_today
                    and not self.overnight_etf_fired
                    and current_time >= t_entry):
                self._step_execute_entries()
            elif (self.scoring_done
                    and not self.entries_done
                    and (self.mr_blocked_today or self.overnight_etf_fired)
                    and current_time >= t_entry):
                logger.info(
                    f"Single-stock MR skipped: mr_blocked_today={self.mr_blocked_today} "
                    f"overnight_etf_fired={self.overnight_etf_fired}"
                )
                self.entries_done = True

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
                    logger.info("Market closed — day complete. Restart at 09:00 tomorrow.")
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

            # Adaptive sleep: 1s during hot windows (open, close, entry
            # times) so transitions are prompt; 30s otherwise so the bot
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

    def _run_failsafe_flatten(self, label: str):
        return morning_exits.run_failsafe_flatten(self, label)


    # ────────────────────────────────────────────────────────────
    # Bar helpers — used by etf_router_runtime.initialize_tape_recording
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _bar_dt(bar: dict) -> Optional[datetime]:
        return etf_router_runtime.bar_dt(bar)

    @staticmethod
    def _bar_float(bar: dict, *keys: str) -> Optional[float]:
        return etf_router_runtime.bar_float(bar, *keys)

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

    def _initialize_tape_recording(self):
        return etf_router_runtime.initialize_tape_recording(self)

    def _update_tape(self, force: bool = False):
        return etf_router_runtime.update_tape(self, force=force)

    def _make_router_decision(self):
        return etf_router_runtime.make_router_decision(self)

    def _make_router_decision_1010(self):
        return etf_router_runtime.make_router_decision_1010(self)

    def _check_etf_exits(self, current_time):
        return etf_router_runtime.check_etf_exits(self, current_time)

    def _execute_etf_exit(self, branch_key: str):
        return etf_router_runtime.execute_etf_exit(self, branch_key)

    # ═══════════════════════════════════════════════════
    # Overnight ETF sleeve delegates
    # ═══════════════════════════════════════════════════
    def _evaluate_overnight_etf_strategies(self):
        return overnight_etf_runner.evaluate_overnight_etf_strategies(self)



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
