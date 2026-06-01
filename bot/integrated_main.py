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
from bot import p1_fallback
from bot import premarket_classifier
from bot import premarket_runner
from bot import scoring
from bot import state_io
from bot import v2_fallback
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
# Unified system timeline: 5:00 start, 10:00 router, 10:05 entry, 10:10-10:30 V2, 10:15 P1
_HOT_WINDOWS_HHMM = (
    (5, 0, 6, 2),     # Premarket dynamic-limit checkpoints (incl. final 06:00 trip)
    (9, 24, 9, 32),   # Order cancel, 09:30 batch sells, 09:31 rescue
    (9, 58, 10, 32),  # ETF tape final, 10:00 decision, 10:05 entry, 10:10-10:30 V2/P1
    (11, 28, 11, 32), # V2 11:30 hybrid checkpoint
    (13, 58, 14, 2),  # SQQQ exit
    (14, 58, 15, 2),  # TQQQ/P1 exit
    (15, 28, 15, 32), # V2 final exit / intraday hard flatten
    (15, 44, 16, 1),  # MR selection, entry, EOD
)


def _is_hot_window(now_t: dt_time) -> bool:
    """True if ``now_t`` falls inside any pre-defined hot window."""
    cur = now_t.hour * 60 + now_t.minute
    for sh, sm, eh, em in _HOT_WINDOWS_HHMM:
        if (sh * 60 + sm) <= cur < (eh * 60 + em):
            return True
    return False


class CombinedOvernightReboundBot:
    """Unified Bot Orchestrator — Intraday ETF (Router > V2 > P1) + Overnight MR
    
    Capital allocation:
    - 90% intraday ETF bucket (Router priority 1, V2 priority 2, P1 priority 3)
    - 30% per MR position, max 3 positions = 90% total, 0.3% ADV cap per symbol
    
    Daily Timeline (ET):
    05:00 - Bot startup, load state, verify calendar
    05:00-06:00 - Premarket MR monitoring and dynamic limit decisions
    09:25 - Cancel premarket limits, prepare for open
    09:30 - Batch market sell remaining MR positions
    09:30-10:00 - Build ETF tape snapshot
    10:00 - Router decision (priority 1)
    10:05 - Router entry (if fired)
    10:10-10:30 - V2 fallback window (priority 2, if router no-trade)
    10:15 - P1 fallback entry (priority 3, if router+V2 no-trade)
    11:30 - V2 hybrid checkpoint (exit/hold decision)
    14:00 - SQQQ Goldilocks exit
    15:00 - Router/P1 TQQQ exit
    15:30 - V2 final exit / hard flatten all intraday ETF
    15:45 - MR candidate selection (if router no-trade and subtype qualifies)
    15:45 - MR batch entry
    16:00 - Market close, final reconciliation
    
    Key rule: Only router trade blocks MR. V2/P1 do NOT block MR.
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
        self.etf_position: Optional[Dict[str, Any]] = None  # Current ETF position if any
        self.etf_opens_930: Dict[str, float] = {}  # 9:30 opens for tape
        self.tape_recording_active = False
        # Throttle for _update_tape (API efficiency #1). Monotonic seconds
        # of the last snapshot fetch; 0.0 means "never run yet".
        self._tape_last_update_monotonic: float = 0.0

        # ═══════════════════════════════════════════════════
        # Unified System: Intraday ETF sleeve state
        # ═══════════════════════════════════════════════════
        # Only one intraday sleeve can be filled per day: Router > V2 > P1
        self.intraday_etf_sleeve_filled = False  # True if any intraday ETF position taken
        
        # Router state extensions
        self.router_entry_pending = False  # Entry queued at 10:00 for 10:05 execution
        self.router_no_trade_subtype: Optional[str] = None  # Classified subtype for V2/P1 eval
        
        # V2 fallback state
        self.v2_active = False
        self.v2_direction: Optional[str] = None  # "long" or "short"
        self.v2_trigger_price: Optional[float] = None
        self.v2_trigger_high: Optional[float] = None
        self.v2_trigger_low: Optional[float] = None
        self.v2_trigger_range_pct: Optional[float] = None
        self.v2_hybrid_checkpoint_done = False  # 11:30 decision made
        # V2 rate-limit protection
        self._v2_last_eval_monotonic: float = 0.0
        
        # P1 fallback state
        self.p1_active = False
        
        # Stage flags
        self.startup_done = False         # 9:00-9:25 startup phase
        self.tape_initialized = False     # 9:30 opens recorded
        self.router_decision_made = False  # 10:00 decision complete
        self.morning_exits_done = False   # All overnight positions exited
        self.data_collected = False       # Universe + daily bars ready
        self.scoring_done = False         # 3:50 PM scoring complete
        self.entries_done = False         # 3:50 PM entries executed

        # Open-exit state
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
        t_failsafe     = _parse_config_time(config.V2_FAILSAFE_TIME)        # 09:45 post-exit failsafe
        t_data_collect = _parse_config_time(config.DATA_COLLECTION_TIME)    # 15:30
        t_scoring      = _parse_config_time(config.SCORING_TIME)            # 15:45
        t_entry        = _parse_config_time(config.ENTRY_TIME)              # 15:45
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

                # 09:25 — cancel any resting premarket limit orders before normal exits.
                # Run regardless of local position state for safety.
                if (not self.morning_open_orders_cancelled
                        and current_time >= t_cancel_orders):
                    logger.warning("09:25 order cleanup: canceling all open orders before 09:30 exits")
                    self.position_mgr.cancel_all_open_orders()
                    self.premarket_limit_order_ids.clear()
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
                    if current_time >= t_exit_all and not getattr(self, "_open_exit_submitted", False):
                        self._submit_open_exit_market_sells()
                        self._open_exit_submitted = True
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
            # UNIFIED SYSTEM: Intraday ETF Sleeve (Router > V2 > P1)
            # 90% of equity, only one sleeve per day, MR not blocked by V2/P1
            # ════════════════════════════════════════════

            if getattr(config, "ETF_ROUTER_ENABLED", False):
                # Unified system timeline constants
                t_startup = _parse_config_time(getattr(config, "BOT_START_TIME", "05:00"))
                t_market_open = _parse_config_time(getattr(config, "MARKET_OPEN_TIME", "09:30"))
                t_router_decision = _parse_config_time(getattr(config, "ROUTER_DECISION_TIME", "10:00"))
                t_router_entry = _parse_config_time(getattr(config, "ROUTER_ENTRY_TIME", "10:05"))
                t_v2_start = _parse_config_time(getattr(config, "V2_LONG_ENTRY_WINDOW_START", "10:10"))
                t_v2_end = _parse_config_time(getattr(config, "V2_LONG_ENTRY_WINDOW_END", "10:30"))
                t_p1_entry = _parse_config_time(getattr(config, "P1_ENTRY_TIME", "10:15"))
                t_v2_checkpoint = _parse_config_time(getattr(config, "V2_HYBRID_CHECKPOINT_TIME", "11:30"))
                t_sqqq_exit = _parse_config_time(getattr(config, "SQQQ_EXIT_TIME", "14:00"))
                t_tqqq_exit = _parse_config_time(getattr(config, "TQQQ_EXIT_TIME", "15:00"))
                t_v2_final_exit = _parse_config_time(getattr(config, "V2_FINAL_EXIT_TIME", "15:30"))
                t_intraday_flatten = _parse_config_time(getattr(config, "INTRADAY_ETF_HARD_FLATTEN_TIME", "15:30"))

                # 05:00 startup - pre-market prep (unified system starts at 5 AM)
                if (not self.startup_done
                        and current_time >= t_startup
                        and current_time < t_market_open):
                    self._run_startup_phase()

                # 09:30 initialize ETF tape (once market opens)
                if (self.startup_done
                        and not self.tape_initialized
                        and not self.router_decision_made
                        and current_time >= t_market_open
                        and current_time < t_router_decision):
                    if current_time > dt_time(9, 31):
                        logger.warning(f"ETF router late-start at {current_time.strftime('%H:%M')} without 09:30 tape; disabling router for today")
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

                # 10:00 make router decision (stores decision, doesn't enter yet)
                if (self.tape_initialized
                        and not self.router_decision_made
                        and current_time >= t_router_decision):
                    self._update_tape(force=True)
                    self._make_router_decision()

                # 10:05 execute router entry (if decision was to trade)
                if (self.router_decision_made
                        and self.router_entry_pending
                        and current_time >= t_router_entry
                        and not self.intraday_etf_sleeve_filled):
                    self._execute_pending_router_entry()

                # ═══════════════════════════════════════════════════
                # Priority 2: V2 Fallback (only if router no-trade)
                # ═══════════════════════════════════════════════════
                if (getattr(config, "ENABLE_V2_FALLBACK", False)
                        and self.router_decision_made
                        and not self.intraday_etf_sleeve_filled
                        and not getattr(self, "router_entry_pending", False)
                        and not self.v2_active):

                    in_v2_window = (
                        current_time >= t_v2_start
                        and current_time <= t_v2_end
                    )

                    if in_v2_window:
                        v2_interval = float(getattr(config, "V2_EVAL_INTERVAL_SECONDS", 10.0))
                        now_mono = time.monotonic()

                        if (now_mono - self._v2_last_eval_monotonic) >= v2_interval:
                            self._v2_last_eval_monotonic = now_mono

                            # Fetch combined snapshots once per V2 tick
                            v2_snaps = self.alpaca.get_snapshots(["QQQ", "SPY", "IWM"]) or {}

                            # Long can evaluate from 10:10 onward.
                            if self._evaluate_v2_long(current_time, snapshots=v2_snaps):
                                self._execute_v2_entry("long", current_time)

                            # Short only starts at configured short window, and only if long did not fill.
                            elif (
                                current_time >= _parse_config_time(getattr(config, "V2_SHORT_ENTRY_WINDOW_START", "10:15"))
                                and not self.intraday_etf_sleeve_filled
                                and not self.v2_active
                            ):
                                if self._evaluate_v2_short(current_time, snapshots=v2_snaps):
                                    self._execute_v2_entry("short", current_time)

                # ═══════════════════════════════════════════════════
                # Priority 3: P1 Fallback (only if router and V2 both no-trade)
                # ═══════════════════════════════════════════════════
                if (getattr(config, "ENABLE_P1_FALLBACK", False)
                        and self.router_decision_made
                        and not self.intraday_etf_sleeve_filled
                        and not self.v2_active):
                    # P1 entry at 10:15 or later (Issue 3: prevent stealing V2 short priority)
                    if current_time >= t_p1_entry and not getattr(self, "p1_entry_checked", False):
                        if self._evaluate_p1_fallback(current_time):
                            self._execute_p1_entry(current_time)
                        self.p1_entry_checked = True

                # ═══════════════════════════════════════════════════
                # V2 Hybrid Checkpoint: 11:30 exit/hold decision
                # ═══════════════════════════════════════════════════
                if (self.v2_active 
                        and not self.v2_hybrid_checkpoint_done
                        and current_time >= t_v2_checkpoint):
                    if self._evaluate_v2_hybrid_checkpoint(current_time):
                        # Checkpoint says exit
                        self._execute_etf_exit()
                    self.v2_hybrid_checkpoint_done = True
                    self._save_state()

                # ═══════════════════════════════════════════════════
                # ETF Exits (uses planned_exit_time from position)
                # ═══════════════════════════════════════════════════
                if self.etf_position:
                    self._check_etf_exits(current_time)

                # ═══════════════════════════════════════════════════
                # V2 Final Exit: 15:30 hard deadline
                # ═══════════════════════════════════════════════════
                if (self.v2_active
                        and current_time >= t_v2_final_exit):
                    logger.warning("V2 final exit reached at 15:30")
                    self._execute_etf_exit()

                # ═══════════════════════════════════════════════════
                # Intraday Hard Flatten: 15:30 - all ETF must be flat
                # ═══════════════════════════════════════════════════
                if (current_time >= t_intraday_flatten
                        and self.etf_position):
                    logger.critical("Intraday hard flatten at 15:30 - forcing ETF exit")
                    self._execute_etf_exit()

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

            # 3:50 PM — Score and rank using 9:30-15:45 bars
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

    def _initialize_tape_recording(self):
        return etf_router_runtime.initialize_tape_recording(self)

    def _update_tape(self, force: bool = False):
        return etf_router_runtime.update_tape(self, force=force)

    def _make_router_decision(self):
        return etf_router_runtime.make_router_decision(self)

    def _check_etf_exits(self, current_time):
        return etf_router_runtime.check_etf_exits(self, current_time)

    def _execute_etf_exit(self):
        return etf_router_runtime.execute_etf_exit(self)

    def _execute_pending_router_entry(self):
        return etf_router_runtime.execute_pending_router_entry(self)

    # ═══════════════════════════════════════════════════
    # Unified System: V2 Fallback delegates
    # ═══════════════════════════════════════════════════
    def _evaluate_v2_long(self, current_time, snapshots=None):
        return v2_fallback.evaluate_v2_long(self, current_time, snapshots=snapshots)

    def _evaluate_v2_short(self, current_time, snapshots=None):
        return v2_fallback.evaluate_v2_short(self, current_time, snapshots=snapshots)

    def _execute_v2_entry(self, direction, current_time):
        return v2_fallback.execute_v2_entry(self, direction, current_time)

    def _evaluate_v2_hybrid_checkpoint(self, current_time):
        return v2_fallback.evaluate_v2_hybrid_checkpoint(self, current_time)

    # ═══════════════════════════════════════════════════
    # Unified System: P1 Fallback delegates
    # ═══════════════════════════════════════════════════
    def _evaluate_p1_fallback(self, current_time):
        return p1_fallback.evaluate_p1_fallback(self, current_time)

    def _execute_p1_entry(self, current_time):
        return p1_fallback.execute_p1_entry(self, current_time)



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
