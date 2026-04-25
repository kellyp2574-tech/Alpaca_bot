"""Overnight Momentum Bot — Main Orchestrator

Daily Schedule (ET) — bot starts at 9:00 AM:

MORNING (T+1 exits — positions from yesterday's 3:50 PM entries):
  09:00  Start, detect overnight positions from broker
  09:30  Market open — capture RTH open prices (no exits; all positions held)
  09:35  Classify each position by entry-price comparison:
           price_935 > entry_price  -> place 1.25% trailing stop (continuation)
           price_935 <= entry_price -> exit immediately at 09:35 (gap faded)
  11:30  Cancel any live trailing stops + market sell remaining positions
  11:35  Post-exit failsafe verification

AFTERNOON (T-1 entries — new positions for tomorrow's exits):
  15:30  Build universe (Massive + Alpaca asset filter + daily bars + ADV)
  15:48  Fetch 9:30-3:50 minute bars -> build & score candidates (350 model)
  15:50  Select positions (account-tier), size, EXECUTE ENTRIES (market)
  16:00  Confirm positions held overnight, save state, done
"""
import logging
import math
import os
import sys
import time
from datetime import datetime, time as dt_time, timedelta, date
from typing import List, Optional, Dict, Any, Tuple
from zoneinfo import ZoneInfo

from bot import config
from bot.massive_client import MassiveClient
from bot.market_data import AlpacaDataClient
from bot.momentum_scorer import (
    MomentumCandidate,
    SelectionConfig,
    get_selection_config,
    Allocation,
    build_signal_candidates_350,
    compute_raw_metrics_350,
    compute_head_score,
    normalize_and_score_350,
    assign_buckets,
    allocate_head_tail,
)
from bot.exit_classifier import (
    classify_positions,
    ExitClassification,
    EXIT_BUCKET_935,
    EXIT_BUCKET_TRAIL,
    EXIT_BUCKET_1130,
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


def _et_utc_offset_str() -> str:
    """Return the current America/New_York UTC offset as an RFC-3339 string.

    Returns "-04:00" during EDT and "-05:00" during EST.
    """
    now_et = datetime.now(_ET)
    offset = now_et.utcoffset()
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    hours, remainder = divmod(abs(total_seconds), 3600)
    minutes = remainder // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def _parse_config_time(time_str: str) -> dt_time:
    """Parse an 'HH:MM' config string into a datetime.time object."""
    parts = time_str.split(":")
    return dt_time(int(parts[0]), int(parts[1]))


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

        # Morning open price capture
        self.hard_stops_checked = False   # flag name kept for state-file compat

        # Exit state
        self.v2_classified = False                    # 9:35 classification done
        self.exit_schedule: Dict[str, str] = {}       # {symbol: exit_bucket}
        self.exits_1130_done = False
        self.classification_audit: Dict[str, dict] = {}  # {symbol: classification metrics}

        # Open prices captured at market open (for V2 move calculation)
        self.open_prices: Dict[str, float] = {}

        # Failsafe
        self.post_exit_failsafe_done = False

        # PDT guard: symbols sold today (no same-day re-entry when equity < $50k)
        self.sold_today: set = set()

        # Trailing stop order IDs placed at 9:35 (for cancel-before-market-sell at 11:30)
        self.trailing_order_ids: Dict[str, str] = {}  # {symbol: order_id}

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
        try:
            self._run()
        except Exception:  # noqa: BLE001
            logger.critical("UNHANDLED EXCEPTION in run() — bot terminated", exc_info=True)
            try:
                self._save_state()
            except Exception:
                logger.critical("State save also failed after crash", exc_info=True)
            raise

    def _run(self):
        """Inner run — called by run() which wraps it with top-level error handling."""
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

        # Pre-compute schedule times from config
        t_market_open  = _parse_config_time(config.MARKET_OPEN_TIME)        # 09:30
        t_v2_classify  = _parse_config_time(config.V2_CLASSIFY_TIME)        # 09:35
        t_bucket_1130  = _parse_config_time(config.EXIT_BUCKET_1130_TIME)   # 11:30
        t_failsafe     = _parse_config_time(config.V2_FAILSAFE_TIME)        # 11:35
        t_data_collect = _parse_config_time(config.DATA_COLLECTION_TIME)    # 15:30
        t_scoring      = _parse_config_time(config.SCORING_TIME)            # 15:48
        t_entry        = _parse_config_time(config.ENTRY_TIME)              # 15:50
        t_market_close = dt_time(16, 0)

        # If starting after failsafe time with positions, flatten immediately
        current_time = datetime.now(_ET).time()
        if current_time >= t_failsafe and self.position_mgr.get_position_count() > 0:
            logger.warning(f"Started after {config.V2_FAILSAFE_TIME} — flattening immediately")
            self._run_failsafe_flatten("late-start flatten")
            self.morning_exits_done = True

        # If starting after 4:00 PM, nothing to do
        if current_time >= t_market_close:
            logger.error("Started after market close — nothing to do")
            return

        # Main event loop
        while True:
            now = datetime.now(_ET)
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
                    # 9:30 AM — Capture RTH open prices (no exits)
                    if not self.hard_stops_checked and current_time >= t_market_open:
                        self._capture_open_prices()
                        self.hard_stops_checked = True
                        self._save_state()

                    # 9:35 AM — Classify + execute immediate 9:35 exits
                    if not self.v2_classified and current_time >= t_v2_classify:
                        self._classify_and_exit_v2()
                        self.v2_classified = True
                        self._save_state()

                    # 11:30 AM — Exit remaining (hold) bucket
                    if self.v2_classified and not self.exits_1130_done and current_time >= t_bucket_1130:
                        self._exit_bucket(EXIT_BUCKET_1130, "scheduled 11:30 AM exit")
                        self.exits_1130_done = True
                        self._save_state()

                    # 11:35 — Post-exit failsafe
                    if not self.post_exit_failsafe_done and current_time >= t_failsafe:
                        bc = self.position_mgr.broker_position_count()
                        if bc > 0:
                            logger.warning(f"Post-exit failsafe: broker still has {bc} positions")
                            self._run_failsafe_flatten(f"{config.V2_FAILSAFE_TIME} post-exit failsafe")
                        elif bc == 0:
                            logger.info("Post-exit failsafe: broker confirmed flat")
                        self.post_exit_failsafe_done = True
                        self.morning_exits_done = True
                        self._save_state()

                    # Early completion — all positions exited before 11:30
                    if (self.v2_classified
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
                if current_time < dt_time(15, 45):
                    self._step_collect_data()
                else:
                    logger.warning("Past 3:45 PM without data collection — attempting now")
                    self._step_collect_data()

            # 3:48 PM — Score and rank (requires data collection)
            if self.data_collected and not self.scoring_done and current_time >= t_scoring:
                self._step_score_and_rank()

            # 3:50 PM — Execute entries (requires scoring)
            if self.scoring_done and not self.entries_done and current_time >= t_entry:
                self._step_execute_entries()

            # ════════════════════════════════════════════
            # Day completion check
            # ════════════════════════════════════════════

            if current_time >= t_market_close:
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

    def _capture_open_prices(self):
        """9:30 AM: Fetch RTH open prices for all positions and store for 9:35 classification.

        No exits are triggered here. All positions are held regardless of the
        opening print. The 9:35 classifier uses these open prices to compute
        ret_open_to_935 and assign each position to the 9:35 or 11:30 bucket.
        """
        logger.info("OPEN PRICE CAPTURE: fetching 9:30 opening prints")

        positions = list(self.position_mgr.positions.items())
        if not positions:
            return

        symbols = [s for s, _ in positions]
        snapshots = self.alpaca.get_snapshots(symbols)

        captured = 0
        for symbol, position in positions:
            snap = snapshots.get(symbol, {})
            open_price = snap.get("open")

            if not open_price or open_price <= 0:
                logger.warning(
                    f"OPEN CAPTURE: {symbol} no snapshot open "
                    f"— will fall back to first minute bar at 9:35"
                )
                continue

            self.open_prices[symbol] = open_price
            position.current_price = open_price
            captured += 1

        logger.info(f"OPEN CAPTURE: {captured}/{len(positions)} open prices captured from snapshot")

    def _classify_and_exit_v2(self):
        """9:35 AM: Classify positions and act:

          price_935 > entry_price  -> place 1.25% trailing stop (continuation)
          price_935 <= entry_price -> exit immediately (gap faded)

        11:30 fallback: cancel any live trailing stops + market sell survivors.
        """
        positions = list(self.position_mgr.positions.items())
        if not positions:
            logger.info("EXIT CLASSIFY: no positions to classify")
            return

        symbols = [s for s, _ in positions]
        logger.info(f"EXIT CLASSIFY: classifying {len(symbols)} positions")

        # 1) Fetch 9:35 snapshots (used as price_935 fallback)
        snapshots = self.alpaca.get_snapshots(symbols)

        # 2) Fetch 9:30-9:35 minute bars (primary source for price_935)
        today = date.today().isoformat()
        et_offset = _et_utc_offset_str()
        minute_bars = self.alpaca.get_minute_bars(
            symbols,
            f"{today}T09:30:00{et_offset}",
            f"{today}T09:35:00{et_offset}",
        )

        # 3) Resolve open prices (for gap_pct logging only — not used for classification)
        open_price_sources: Dict[str, str] = {}
        for symbol in symbols:
            if symbol in self.open_prices:
                open_price_sources[symbol] = "snapshot"
            else:
                bars = minute_bars.get(symbol, [])
                if bars:
                    bar_open = bars[0].get("o")
                    if bar_open and bar_open > 0:
                        self.open_prices[symbol] = bar_open
                        open_price_sources[symbol] = "minute_bar"
                    else:
                        open_price_sources[symbol] = "missing"
                else:
                    open_price_sources[symbol] = "missing"

        # 4) Build entry_prices for classification
        entry_prices = {s: p.entry_price for s, p in positions}

        # 5) Classify all positions
        classifications = classify_positions(
            symbols=symbols,
            open_prices=self.open_prices,
            open_price_sources=open_price_sources,
            snapshots_935=snapshots,
            minute_bars_935=minute_bars,
            entry_prices=entry_prices,
        )

        # 6) Store schedule and classification audit
        self.exit_schedule = {sym: cls.exit_bucket for sym, cls in classifications.items()}
        self.classification_audit = {
            sym: {
                "entry_price":       cls.entry_price,
                "open_price":        cls.open_price,
                "price_935":         cls.price_935,
                "ret_vs_entry_pct":  cls.ret_vs_entry_pct,
                "exit_bucket":       cls.exit_bucket,
                "open_price_source": cls.open_price_source,
                "price_935_source":  cls.price_935_source,
                "gap_pct":           cls.gap_pct,
            }
            for sym, cls in classifications.items()
        }

        # 7) Execute immediate exits for faded positions
        exits_935 = [sym for sym, cls in classifications.items()
                     if cls.exit_bucket == EXIT_BUCKET_935]
        if exits_935:
            logger.info(f"EXIT CLASSIFY: {len(exits_935)} positions faded — exiting immediately")
            for symbol in exits_935:
                self._exit_single_position(symbol, "price_935 <= entry_price -> 9:35 exit")
        else:
            logger.info("EXIT CLASSIFY: no positions in immediate-exit bucket")

        # 8) Place trailing stops for continuation positions
        trail_symbols = [sym for sym, cls in classifications.items()
                         if cls.exit_bucket == EXIT_BUCKET_TRAIL
                         and sym in self.position_mgr.positions]
        if trail_symbols:
            logger.info(
                f"EXIT CLASSIFY: {len(trail_symbols)} continuation positions — "
                f"placing {config.TRAILING_STOP_PCT}% trailing stops"
            )
            for symbol in trail_symbols:
                position = self.position_mgr.positions.get(symbol)
                if not position:
                    continue
                trail_resp = self.position_mgr.submit_trailing_stop_sell(
                    symbol=symbol,
                    qty=position.quantity,
                    trail_percent=config.TRAILING_STOP_PCT,
                )
                if trail_resp:
                    order_id = trail_resp.get("id")
                    if order_id:
                        self.trailing_order_ids[symbol] = order_id
                        logger.info(
                            f"TRAIL {symbol}: trailing stop placed "
                            f"(qty={position.quantity}, trail={config.TRAILING_STOP_PCT}%, "
                            f"order_id={order_id})"
                        )
                else:
                    logger.warning(
                        f"TRAIL {symbol}: failed to place trailing stop — "
                        f"will fallback to market sell at 11:30"
                    )
        else:
            logger.info("EXIT CLASSIFY: no positions in trailing-stop bucket")

    def _exit_bucket(self, bucket: str, reason: str):
        """Exit all positions in the given bucket.

        For the 11:30 bucket this also cancels any live trailing stop orders
        before submitting market sells, to avoid double-selling.
        For any remaining positions not in the schedule (e.g. trailing stop
        already filled them), broker reconciliation will clean up local state.
        """
        # Determine symbols: include scheduled-for-this-bucket AND any
        # unscheduled survivors still held locally (catch-all).
        if bucket == EXIT_BUCKET_1130:
            # Cancel all live trailing stops first
            trail_symbols_to_cancel = [
                sym for sym in list(self.trailing_order_ids.keys())
                if sym in self.position_mgr.positions
            ]
            if trail_symbols_to_cancel:
                logger.info(
                    f"{reason}: canceling {len(trail_symbols_to_cancel)} live trailing stops "
                    f"before market sell: {trail_symbols_to_cancel}"
                )
                for symbol in trail_symbols_to_cancel:
                    order_id = self.trailing_order_ids.get(symbol)
                    if order_id:
                        self.position_mgr._cancel_order(order_id)
                    self.trailing_order_ids.pop(symbol, None)

            # Exit ALL remaining local positions (trailing stop may have filled some)
            # First reconcile so we don't try to sell already-closed positions
            self.position_mgr.reconcile_local_positions_from_broker()
            symbols = list(self.position_mgr.positions.keys())
        else:
            symbols = [sym for sym, t in self.exit_schedule.items()
                       if t == bucket and sym in self.position_mgr.positions]

        if not symbols:
            logger.info(f"{reason}: no positions remaining")
            return

        logger.info(f"{reason}: exiting {len(symbols)} positions: {symbols}")
        for symbol in symbols:
            self._exit_single_position(symbol, reason)

        # Reconcile local state with broker after exit wave
        actions = self.position_mgr.reconcile_local_positions_from_broker()
        if actions:
            logger.info(f"{reason}: post-exit reconciliation adjustments: {actions}")
            self._save_state()

        remaining = self.position_mgr.get_position_count()
        logger.info(f"{reason}: done — {remaining} total positions remaining")

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
            current_bucket = self.exit_schedule.get(symbol)
            # Both 9:35 exits and incomplete trailing-stop exits fall back to 11:30
            if current_bucket in (EXIT_BUCKET_935, EXIT_BUCKET_TRAIL):
                self.exit_schedule[symbol] = EXIT_BUCKET_1130
                logger.warning(
                    f"EXIT INCOMPLETE {symbol}: {remaining} shares still held — "
                    f"re-routed {current_bucket} -> {EXIT_BUCKET_1130}"
                )
            else:
                logger.warning(
                    f"EXIT INCOMPLETE {symbol}: {remaining} shares still held in "
                    f"last bucket ({current_bucket}) — failsafe will catch"
                )
        self._save_state()

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
            logger.info(f"Stage C data quality: {pre_c_count} -> {len(quality_passed)}")
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

            # 5b. Compute HEAD score (late-day continuation signal)
            candidates = compute_head_score(candidates, self._minute_bars)

            # 6. Normalize, score, bucket
            candidates = normalize_and_score_350(candidates)
            candidates = assign_buckets(candidates)
            candidates.sort(key=lambda c: c.composite_score, reverse=True)

            self.scored_candidates = candidates
            self.scoring_done = True
            self._save_state()

            # Log top 10
            logger.info(f"Scoring complete: {len(candidates)} scored")
            head_sorted = sorted(candidates, key=lambda c: c.head_score, reverse=True)
            logger.info("Top 10 by HEAD score:")
            for c in head_sorted[:10]:
                logger.info(
                    f"  {c.symbol}: head={c.head_score:.4f} score={c.composite_score:.3f} "
                    f"bucket={c.bucket} ret={c.intraday_return:.2%} "
                    f"prox={c.proximity_to_high:.3f} atr%={c.atr_percent:.3f}"
                )

            # Save candidates audit artifact — two ranked views
            def _candidate_dict(c):
                return {
                    "symbol": c.symbol,
                    "score": round(c.composite_score, 4),
                    "head_score": round(c.head_score, 4),
                    "bucket": c.bucket,
                    "intraday_return": round(c.intraday_return, 4),
                    "proximity_to_high": round(c.proximity_to_high, 4),
                    "volume_vs_avg": round(c.volume_vs_avg, 2),
                    "volume_trend": round(c.volume_trend, 2),
                    "vs_market": round(c.vs_market, 4),
                    "atr_percent": round(c.atr_percent, 4),
                    "signal_price": round(c.signal_price, 4),
                    "adv_dollars": round(c.adv_dollars, 0),
                }

            top_20_head = sorted(candidates, key=lambda c: c.head_score, reverse=True)[:20]
            top_20_tail = sorted(candidates, key=lambda c: c.composite_score, reverse=True)[:20]
            audit_dicts = {
                "top_20_by_head_score": [_candidate_dict(c) for c in top_20_head],
                "top_20_by_composite_score": [_candidate_dict(c) for c in top_20_tail],
            }
            save_candidates_audit(audit_dicts)

            # Also update universe audit with top 20
            if self._universe_diag:
                save_universe_audit(self._universe_diag, self.universe, scored_top20=[_candidate_dict(c) for c in top_20_head])

        except Exception as e:
            logger.exception(f"Error in scoring: {e}")
            self.scoring_done = True

    def _step_execute_entries(self):
        """3:50 PM: HEAD/TAIL allocation -> execution-gate -> market buys."""
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

            # Get account equity for tier selection + PDT check
            equity = self.position_mgr.get_account_equity()
            if not equity or equity <= 0:
                logger.error("Cannot determine account equity — skipping entries")
                self.entries_done = True
                return

            # Use buying power as deployable capital so margin is used naturally
            buying_power = self.position_mgr.get_total_capital()
            if not buying_power or buying_power <= 0:
                logger.warning("Cannot determine buying power — falling back to equity")
                buying_power = equity

            sel = get_selection_config(equity)
            max_deployable = equity * sel.max_leverage
            deployable = min(buying_power, max_deployable)
            logger.info(
                f"Account equity: ${equity:,.2f}, buying_power: ${buying_power:,.2f}, "
                f"max_leverage={sel.max_leverage}x, deployable: ${deployable:,.2f}"
            )

            # Fix 5: PDT-aware allocation - filter candidates BEFORE allocation
            if equity < 50_000 and self.sold_today:
                before = len(self.scored_candidates)
                eligible_candidates = [c for c in self.scored_candidates if c.symbol not in self.sold_today]
                blocked = before - len(eligible_candidates)
                if blocked:
                    logger.warning(
                        f"PDT guard: filtered out {blocked} same-day re-entry candidates "
                        f"before allocation (equity ${equity:,.0f} < $50k, sold_today={self.sold_today})"
                    )
                if not eligible_candidates:
                    logger.warning("No candidates remaining after PDT filter — skipping entries")
                    self.entries_done = True
                    return
            else:
                eligible_candidates = self.scored_candidates

            # HEAD/TAIL allocation (tier-aware) on PDT-filtered candidates
            allocations = allocate_head_tail(eligible_candidates, deployable, sel=sel)

            if not allocations:
                logger.warning("No positions allocated — skipping entries")
                self.entries_done = True
                return

            exec_diag.selected_symbols = [a.symbol for a in allocations]

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

            # Submit market buy orders (BP-aware + resize-on-reject + logging)
            total_deployed = 0.0

            def _adaptive_qty(alloc: Allocation, bp_buffer: float = config.ENTRY_BP_BUFFER_PCT) -> int:
                """Return alloc.shares clamped to current buying power."""
                bp = self.position_mgr.get_total_capital()
                if not bp or bp <= 0:
                    return alloc.shares
                max_notional = bp * bp_buffer
                price_ref = alloc.candidate.signal_price
                notional = alloc.shares * price_ref
                if notional <= max_notional:
                    return alloc.shares
                new_qty = math.floor(max_notional / price_ref) if price_ref > 0 else 0
                return max(0, new_qty)

            for alloc in allocations:
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
                    f"bucket={alloc.alloc_bucket}, rank={alloc.rank}"
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
                    # Retry once with fresh BP + recalculated qty
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

                fill = self.position_mgr.get_order_fill(order_id, max_wait=30)
                if fill and int(fill["filled_qty"]) > 0:
                    filled_qty = int(fill["filled_qty"])
                    fill_price = fill["filled_avg_price"]

                    position = Position(
                        symbol=symbol,
                        entry_price=fill_price,
                        quantity=filled_qty,
                        entry_time=datetime.now(_ET),
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
                        "alloc_bucket": alloc.alloc_bucket,
                        "rank": alloc.rank,
                    }

                    logger.info(
                        f"ENTRY {symbol}: {filled_qty} @ {fill_price:.4f} "
                        f"[{alloc.alloc_bucket} #{alloc.rank}] "
                        f"(score={candidate.composite_score:.3f}, bucket={candidate.bucket})"
                    )
                else:
                    # Cancel potentially-live order so mop-up can safely retry
                    self.position_mgr._cancel_order(order_id)
                    logger.warning(f"No fill for {symbol} buy order (order canceled)")
                    exec_diag.failed_submissions[symbol] = "no_fill"

            # ── Post-loop mop-up pass ────────────────────────────
            # Only filled symbols are truly "done". Failed submissions
            # can be retried with smaller qty, and candidates outside
            # the original allocation can be vet-on-demand.
            deploy_target = deployable * config.ENTRY_MIN_DEPLOY_PCT
            if total_deployed < deploy_target:
                shortfall = deploy_target - total_deployed
                logger.warning(
                    f"MOP-UP: deployment ${total_deployed:,.2f} < target ${deploy_target:,.2f} "
                    f"(shortfall ${shortfall:,.2f}). Walking remaining candidates..."
                )

                already_filled = set(exec_diag.filled_symbols)
                mopup_attempts = 0
                mopup_fills = 0

                for candidate in eligible_candidates:
                    if mopup_attempts >= config.ENTRY_MOPUP_MAX_POSITIONS:
                        break
                    if len(exec_diag.filled_symbols) >= sel.max_positions:
                        logger.info(f"MOP-UP STOP: reached max_positions ({sel.max_positions})")
                        break
                    sym = candidate.symbol
                    if sym in already_filled:
                        continue

                    price_ref = candidate.signal_price
                    if price_ref <= 0:
                        continue

                    # Vet new candidates not in the original orderable_set
                    if sym not in orderable_set:
                        try:
                            snap = self.alpaca.get_snapshots([sym])
                            ok, rej = filter_execution_ready(
                                [sym], snap,
                                max_spread_pct=0.05, require_quote=True,
                            )
                            if ok:
                                orderable_set.add(sym)
                                logger.info(f"MOP-UP VET {sym}: passed execution gate")
                            else:
                                reason = rej.get(sym, "unknown")
                                logger.info(f"MOP-UP VET {sym}: rejected ({reason})")
                                continue
                        except Exception as e:
                            logger.warning(f"MOP-UP VET {sym}: snapshot error ({e})")
                            continue

                    fresh_bp = self.position_mgr.get_total_capital()
                    if not fresh_bp or fresh_bp <= 0:
                        break

                    # Per-position cap: ADV, MAX_POSITION_DOLLARS, and BP share
                    adv_cap = candidate.adv_dollars * config.ADV_CAP_PCT if candidate.adv_dollars > 0 else fresh_bp
                    target_dollars = min(
                        fresh_bp * config.ENTRY_BP_BUFFER_PCT,
                        adv_cap,
                        config.MAX_POSITION_DOLLARS,
                    )
                    if target_dollars < price_ref * config.MIN_SHARES:
                        logger.info(
                            f"MOP-UP SKIP {sym}: capped target ${target_dollars:,.2f} "
                            f"can't support min shares at price {price_ref:.4f}"
                        )
                        continue

                    mopup_qty = math.floor(target_dollars / price_ref)
                    if mopup_qty < config.MIN_SHARES:
                        continue

                    mopup_attempts += 1

                    logger.info(
                        f"MOP-UP PLANNED {sym}: qty={mopup_qty}, price_ref={price_ref:.4f}, "
                        f"notional={mopup_qty * price_ref:,.2f}, bp_before={fresh_bp:,.2f}, "
                        f"attempt={mopup_attempts}/{config.ENTRY_MOPUP_MAX_POSITIONS}"
                    )

                    buy_resp = self.position_mgr.submit_buy_order(sym, mopup_qty)
                    if not buy_resp:
                        continue

                    order_id = buy_resp.get("id")
                    if not order_id:
                        continue

                    fill = self.position_mgr.get_order_fill(order_id, max_wait=30)
                    if fill and int(fill["filled_qty"]) > 0:
                        filled_qty = int(fill["filled_qty"])
                        fill_price = fill["filled_avg_price"]

                        position = Position(
                            symbol=sym,
                            entry_price=fill_price,
                            quantity=filled_qty,
                            entry_time=datetime.now(_ET),
                            entry_gap_pct=0.0,
                            adv_estimate=candidate.adv_dollars,
                            peak_price=fill_price,
                            current_price=fill_price,
                        )
                        self.position_mgr.positions[sym] = position
                        total_deployed += fill_price * filled_qty
                        exec_diag.filled_symbols.append(sym)
                        exec_diag.fill_details[sym] = {
                            "qty": filled_qty, "price": round(fill_price, 4),
                            "score": round(candidate.composite_score, 4),
                            "bucket": candidate.bucket,
                            "alloc_bucket": "MOPUP",
                            "rank": 0,
                        }
                        already_filled.add(sym)
                        mopup_fills += 1

                        logger.info(
                            f"ENTRY MOP-UP {sym}: {filled_qty} @ {fill_price:.4f} "
                            f"(score={candidate.composite_score:.3f})"
                        )
                    else:
                        self.position_mgr._cancel_order(order_id)
                        already_filled.add(sym)
                        logger.warning(f"No fill for mop-up {sym} (order canceled)")

                logger.info(
                    f"MOP-UP complete: {mopup_fills} fills from {mopup_attempts} attempts, "
                    f"total deployed now ${total_deployed:,.2f}"
                )

            self.entries_done = True
            self._save_state()

            # Store execution stats for health report
            head_filled = sum(1 for s in exec_diag.filled_symbols
                              if exec_diag.fill_details.get(s, {}).get("alloc_bucket") == "HEAD")
            tail_filled = sum(1 for s in exec_diag.filled_symbols
                              if exec_diag.fill_details.get(s, {}).get("alloc_bucket") == "TAIL")
            mopup_filled = sum(1 for s in exec_diag.filled_symbols
                               if exec_diag.fill_details.get(s, {}).get("alloc_bucket") == "MOPUP")

            self._exec_stats = {
                "selected": len(exec_diag.selected_symbols),
                "orderable": len(exec_diag.orderable_symbols),
                "exec_rejected": len(exec_diag.rejected_symbols),
                "exec_rejected_reasons": exec_diag.rejected_symbols,
                "orders_submitted": len(exec_diag.submitted_symbols),
                "entries_filled": len(exec_diag.filled_symbols),
                "head_filled": head_filled,
                "tail_filled": tail_filled,
                "mopup_filled": mopup_filled,
                "total_deployed": total_deployed,
                "equity": equity,
                "deployable": deployable,
            }

            deployment_pct = total_deployed / deployable * 100 if deployable > 0 else 0.0
            logger.info(
                f"Entry execution complete: {len(exec_diag.filled_symbols)} filled "
                f"({head_filled} HEAD + {tail_filled} TAIL + {mopup_filled} MOPUP), "
                f"{len(exec_diag.rejected_symbols)} rejected at execution gate, "
                f"${total_deployed:,.2f} deployed "
                f"({deployment_pct:.1f}% of deployable)"
            )

            # Explicit shortfall diagnostics
            if deployment_pct < 80.0:
                logger.warning("=== DEPLOYMENT SHORTFALL DIAGNOSTICS ===")
                
                # PDT blocks (already filtered at allocation stage)
                if equity < 50_000 and self.sold_today:
                    blocked_by_pdt = len(self.sold_today.intersection(set([c.symbol for c in self.scored_candidates])))
                    logger.warning(f"PDT guard blocked {blocked_by_pdt} candidates at allocation stage (equity < $50k)")

                # Execution gate rejections
                if exec_diag.rejected_symbols:
                    reasons = {}
                    for sym, reason in exec_diag.rejected_symbols.items():
                        reasons.setdefault(reason, []).append(sym)
                    for reason, syms in reasons.items():
                        logger.warning(f"Execution gate rejected {len(syms)} symbols: {reason} (e.g., {syms[:3]})")

                # Failed submissions/no fills
                if exec_diag.failed_submissions:
                    failed_reasons = {}
                    for sym, reason in exec_diag.failed_submissions.items():
                        failed_reasons.setdefault(reason, []).append(sym)
                    for reason, syms in failed_reasons.items():
                        logger.warning(f"Failed submissions: {len(syms)} symbols ({reason}) (e.g., {syms[:3]})")

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
                "date": datetime.now(_ET).strftime("%Y-%m-%d"),
                "morning_exits_done": self.morning_exits_done,
                "hard_stops_checked": self.hard_stops_checked,
                "v2_classified": self.v2_classified,
                "exit_schedule": self.exit_schedule,
                "exits_1130_done": self.exits_1130_done,
                "classification_audit": self.classification_audit,
                "post_exit_failsafe_done": self.post_exit_failsafe_done,
                "data_collected": self.data_collected,
                "scoring_done": self.scoring_done,
                "entries_done": self.entries_done,
                "open_prices": self.open_prices,
                "sold_today": list(self.sold_today),
                "trailing_order_ids": self.trailing_order_ids,
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
        self.hard_stops_checked = bot_state.get("hard_stops_checked", False)
        self.v2_classified = bot_state.get("v2_classified", False)
        self.exit_schedule = bot_state.get("exit_schedule", {})
        self.exits_1130_done = bot_state.get("exits_1130_done", False)
        self.classification_audit = bot_state.get("classification_audit", {})
        self.post_exit_failsafe_done = bot_state.get("post_exit_failsafe_done", False)
        self.data_collected = bot_state.get("data_collected", False)
        self.scoring_done = bot_state.get("scoring_done", False)
        self.entries_done = bot_state.get("entries_done", False)
        self.open_prices = bot_state.get("open_prices", {})
        self.sold_today = set(bot_state.get("sold_today", []))
        self.trailing_order_ids = bot_state.get("trailing_order_ids", {})

        # Load positions
        saved = self.state_mgr.load_positions()
        if saved:
            self.position_mgr.load_positions(saved)
            logger.info(f"Loaded {len(saved)} saved positions")


def main():
    try:
        bot = OvernightMomentumBot()
    except Exception:
        logging.critical("UNHANDLED EXCEPTION during bot initialisation", exc_info=True)
        raise
    bot.run()


if __name__ == "__main__":
    main()
