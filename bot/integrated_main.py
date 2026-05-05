"""Combined Overnight Rebound Bot — Main Orchestrator

Sleeve 1: MR_WIDE (Mean Reversion)
  - Buy 15:55, $1–5, day_ret <= -3%, vol_ratio >= 1.5x, close_position <= 0.20
  - Exit 09:35

Sleeve 2: GDP_BASE (Green-Day Pullback)
  - Buy 15:55, $1–10, day_ret +1% to +10%, below VWAP, late_mom <= 0
  - Exit 09:35

Allocation: 60/40 MR/GDP for paper trading (12 MR slots, 8 GDP slots)

Daily Schedule (ET) — bot starts at 9:00 AM:

MORNING (T+1 exits — positions from yesterday's 15:55 entries):
  09:00  Start, detect overnight positions from broker
  09:30  Market open — positions fill at the open
  09:35  Market sell ALL positions (both GDP and MR sleeves)
  09:45  Post-exit failsafe verification

AFTERNOON (T-1 entries — new positions for tomorrow's exits):
  15:30  Build universe (Massive + Alpaca, $1–10, ADV sizing cap protects)
  15:55  Fetch latest 9:30-15:55 minute bars, build both MR and GDP candidates
  15:55  Execute entries immediately after scoring
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
        self.scoring_done = False         # 3:53 PM scoring complete
        self.entries_done = False         # 3:55 PM entries executed

        # Sleeve-specific exit flags
        self.gdp_exits_done = False
        self.mr_exits_done = False

        # Failsafe
        self.post_exit_failsafe_done = False

        # PDT guard: symbols sold today (no same-day re-entry when equity < $50k)
        self.sold_today: set = set()

        # Retry counters
        self.universe_retry_count = 0

        # Data collection results (stored between steps)
        self._minute_bars: Dict[str, List[dict]] = {}
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
        logger.info("Combined Overnight Rebound Bot Starting")
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
        t_exit_all     = _parse_config_time(config.GDP_EXIT_TIME)           # 09:35 (both sleeves)
        t_failsafe     = _parse_config_time(config.V2_FAILSAFE_TIME)        # 09:45
        t_data_collect = _parse_config_time(config.DATA_COLLECTION_TIME)    # 15:30
        t_scoring      = _parse_config_time(config.SCORING_TIME)            # 15:53
        t_entry        = _parse_config_time(config.ENTRY_TIME)              # 15:55
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
                        # Update has_positions so exits can run immediately in this loop
                        has_positions = self.position_mgr.get_position_count() > 0

                if has_positions and not self.morning_exits_done:
                    # 9:35 AM — Exit ALL positions (both GDP and MR sleeves)
                    if not self.gdp_exits_done and current_time >= t_exit_all:
                        self._exit_sleeve_positions("GDP", "09:35 all positions (GDP)")
                        self._exit_sleeve_positions("MR", "09:35 all positions (MR)")
                        self.gdp_exits_done = True
                        self.mr_exits_done = True
                        self._save_state()

                    # 9:45 AM — Post-exit failsafe
                    if (self.gdp_exits_done and self.mr_exits_done
                            and not self.post_exit_failsafe_done and current_time >= t_failsafe):
                        bc = self.position_mgr.broker_position_count()
                        if bc > 0:
                            logger.warning(f"Post-exit failsafe: broker still has {bc} positions")
                            self._run_failsafe_flatten(f"{config.V2_FAILSAFE_TIME} post-exit failsafe")
                        elif bc == 0:
                            logger.info("Post-exit failsafe: broker confirmed flat")
                        self.post_exit_failsafe_done = True
                        self.morning_exits_done = True
                        self._save_state()

                    # Early completion — all positions exited before 9:45
                    if (self.gdp_exits_done and self.mr_exits_done
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

            # 3:55 PM — Score and rank using latest available bars
            if self.data_collected and not self.scoring_done and current_time >= t_scoring:
                self._step_score_and_rank()

            # 3:55 PM — Execute entries (requires scoring)
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

    def _exit_sleeve_positions(self, sleeve: str, reason: str):
        """Exit all positions of a specific sleeve at market.

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
        for symbol in positions:
            self._exit_single_position(symbol, reason)

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
                f"failsafe will catch at {config.V2_FAILSAFE_TIME}"
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
        """~3:55 PM: Fetch latest 9:30-15:55 bars, build both MR and GDP candidates, filter."""
        logger.info("=" * 50)
        logger.info("SCORING (combined): Building MR and GDP candidates")
        logger.info("=" * 50)

        try:
            today = date.today().isoformat()
            signal_end = config.ENTRY_TIME  # 15:55

            # 1. Fetch 9:30-15:55 minute bars for the full base universe
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
            self.mr_candidates = filtered_mr[:config.MR_MAX_POSITIONS]

            # CRITICAL DIAGNOSTIC: MR pipeline counts
            logger.info(
                f"MR pipeline: "
                f"universe={len(self.universe)}, "
                f"raw={len(raw_mr)}, "
                f"passed={len(filtered_mr)}, "
                f"selected={len(self.mr_candidates)}"
            )

            # 4. Build raw GDP candidates from minute bars
            raw_gdp = build_green_day_pullback_candidates(
                self.universe,
                self._minute_bars,
                self._adv_cache,
            )
            filtered_gdp = filter_green_day_pullback_candidates(raw_gdp)

            # 5. Remove GDP candidates that are already MR candidates (MR takes priority)
            mr_symbols = {c.symbol for c in self.mr_candidates}
            filtered_gdp = [c for c in filtered_gdp if c.symbol not in mr_symbols]
            self.gdp_candidates = filtered_gdp[:config.GDP_MAX_POSITIONS]

            # CRITICAL DIAGNOSTIC: GDP pipeline counts
            logger.info(
                f"GDP pipeline: "
                f"universe={len(self.universe)}, "
                f"raw={len(raw_gdp)}, "
                f"passed={len(filtered_gdp)}, "
                f"selected={len(self.gdp_candidates)}"
            )

            self.scoring_done = True
            self._save_state()

            logger.info(
                f"Combined scoring: MR raw={len(raw_mr)} passed={len(filtered_mr)} selected={len(self.mr_candidates)} | "
                f"GDP raw={len(raw_gdp)} passed={len(filtered_gdp)} selected={len(self.gdp_candidates)}"
            )

            # Log MR candidates
            for c in self.mr_candidates:
                logger.info(
                    f"MR SELECT {c.symbol}: price={c.signal_price:.2f}, "
                    f"day_ret={c.day_return:.2%}, vol_ratio={c.volume_ratio:.2f}x, "
                    f"close_pos={c.close_position:.2f}, "
                    f"late_drop={c.late_drop_1530_1550:.2%}, "
                    f"score={c.selection_score:.3f}"
                )

            # Log GDP candidates
            for c in self.gdp_candidates:
                logger.info(
                    f"GDP SELECT {c.symbol}: price={c.signal_price:.2f}, "
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
                "mr_selected": [_mr_dict(c) for c in self.mr_candidates],
                "mr_all_passed": [_mr_dict(c) for c in filtered_mr],
                "gdp_selected": [_gdp_dict(c) for c in self.gdp_candidates],
                "gdp_all_passed": [_gdp_dict(c) for c in filtered_gdp],
            }
            save_candidates_audit(audit_dicts)

            if self._universe_diag:
                top_mr = [_mr_dict(c) for c in filtered_mr[:20]]
                top_gdp = [_gdp_dict(c) for c in filtered_gdp[:20]]
                save_universe_audit(
                    self._universe_diag, self.universe,
                    scored_top20=top_mr + top_gdp,
                )

        except Exception as e:
            logger.exception(f"Error in scoring: {e}")
            self.scoring_done = True

    def _step_execute_entries(self):
        """3:55 PM: Dual-sleeve allocation (60/40 MR/GDP budget) -> execution-gate -> market buys."""
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
            rank: int
            sleeve: str
            candidate: Any

        def _size_candidate(c, slot_dollars: float) -> int:
            """Return share count for a candidate given slot dollars."""
            if c.signal_price <= 0:
                return 0
            adv_cap = c.adv_dollars * config.ADV_CAP_PCT if c.adv_dollars > 0 else slot_dollars
            target_dollars = min(slot_dollars, adv_cap, config.MAX_POSITION_DOLLARS)
            return math.floor(target_dollars / c.signal_price)

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
            mr_slot = mr_budget / config.MR_MAX_POSITIONS if config.MR_MAX_POSITIONS > 0 else 0
            gdp_slot = gdp_budget / config.GDP_MAX_POSITIONS if config.GDP_MAX_POSITIONS > 0 else 0

            logger.info(
                f"Sleeve budgets: MR ${mr_budget:,.2f} ({config.MR_ALLOCATION_PCT:.0%}, "
                f"slot=${mr_slot:,.2f} x {config.MR_MAX_POSITIONS}) | "
                f"GDP ${gdp_budget:,.2f} ({config.GDP_ALLOCATION_PCT:.0%}, "
                f"slot=${gdp_slot:,.2f} x {config.GDP_MAX_POSITIONS})"
            )

            # Build allocations for both sleeves (track min-share skips)
            allocations: List[SleeveAllocation] = []
            mr_min_share_skips = 0
            gdp_min_share_skips = 0

            for rank, c in enumerate(self.mr_candidates[:config.MR_MAX_POSITIONS], start=1):
                shares = _size_candidate(c, mr_slot)
                if shares >= config.MIN_SHARES:
                    allocations.append(SleeveAllocation(c.symbol, shares, rank, "MR", c))
                else:
                    mr_min_share_skips += 1
                    logger.warning(f"MR SKIP {c.symbol}: shares {shares} < min {config.MIN_SHARES}")

            for rank, c in enumerate(self.gdp_candidates[:config.GDP_MAX_POSITIONS], start=1):
                shares = _size_candidate(c, gdp_slot)
                if shares >= config.MIN_SHARES:
                    allocations.append(SleeveAllocation(c.symbol, shares, rank, "GDP", c))
                else:
                    gdp_min_share_skips += 1
                    logger.warning(f"GDP SKIP {c.symbol}: shares {shares} < min {config.MIN_SHARES}")

            # Log min-share skip summary
            if mr_min_share_skips or gdp_min_share_skips:
                logger.warning(
                    f"Min-share skips: MR={mr_min_share_skips}, GDP={gdp_min_share_skips}, "
                    f"MIN_SHARES={config.MIN_SHARES}"
                )

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
            logger.info(f"Selected {len(allocations)} allocations: "
                        f"{sum(1 for a in allocations if a.sleeve=='MR')} MR + "
                        f"{sum(1 for a in allocations if a.sleeve=='GDP')} GDP")

            # Execution eligibility gate
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

            # Submit market buy orders
            total_deployed = 0.0

            def _adaptive_qty(alloc: SleeveAllocation, bp_buffer: float = config.ENTRY_BP_BUFFER_PCT) -> int:
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

            # Hard cutoff for new buy submissions (don't chase too close to close)
            entry_cutoff = dt_time(15, 58, 30)

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

            # Mop-up disabled for paper trading (ENTRY_MOPUP_MAX_POSITIONS = 0)
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
                "mr_selected": [_mr_dict(c) for c in self.mr_candidates],
                "gdp_selected": [_gdp_dict(c) for c in self.gdp_candidates],
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
