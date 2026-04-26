"""Overnight Momentum Bot — Main Orchestrator

Daily Schedule (ET) — bot starts at 9:00 AM:

MORNING (T+1 exits — positions from yesterday's 3:50 PM entries):
  09:00  Start, detect overnight positions from broker
  09:30  Market open — positions fill at the open
  09:40  Market sell ALL positions — no conditions, no trailing stop
  09:45  Post-exit failsafe verification

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
from datetime import datetime, time as dt_time, date
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

        # Exit state (v2_classified reused as "9:40 sell fired" flag)
        self.v2_classified = False

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
        t_exit_940     = _parse_config_time(config.EXIT_940_TIME)           # 09:40
        t_failsafe     = _parse_config_time(config.V2_FAILSAFE_TIME)        # 09:45
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
                    # 9:40 AM — Market sell ALL positions, no conditions
                    if not self.v2_classified and current_time >= t_exit_940:
                        self._exit_all_940()
                        self.v2_classified = True
                        self._save_state()

                    # 9:45 AM — Post-exit failsafe
                    if self.v2_classified and not self.post_exit_failsafe_done and current_time >= t_failsafe:
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

    def _exit_all_940(self):
        """9:40 AM: Market sell ALL positions unconditionally.

        No classification. No trailing stops. Every position is sold at market.
        """
        positions = list(self.position_mgr.positions.keys())
        if not positions:
            logger.info("EXIT 9:40: no positions to sell")
            return

        logger.info(f"EXIT 9:40: market selling {len(positions)} positions: {positions}")
        for symbol in positions:
            self._exit_single_position(symbol, "9:40 AM market sell")

        # Reconcile local state with broker
        actions = self.position_mgr.reconcile_local_positions_from_broker()
        if actions:
            logger.info(f"EXIT 9:40: post-exit reconciliation adjustments: {actions}")

        remaining = self.position_mgr.get_position_count()
        logger.info(f"EXIT 9:40: done — {remaining} positions remaining")

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
                        adv_estimate=candidate.adv_dollars,
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

                    # Per-position cap: ADV and BP share
                    adv_cap = candidate.adv_dollars * config.ADV_CAP_PCT if candidate.adv_dollars > 0 else fresh_bp
                    target_dollars = min(
                        fresh_bp * config.ENTRY_BP_BUFFER_PCT,
                        adv_cap,
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
                            adv_estimate=candidate.adv_dollars,
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
                def _candidate_dict(c):
                    return {
                        "symbol": c.symbol,
                        "score": round(c.composite_score, 4),
                        "head_score": round(c.head_score, 4),
                        "bucket": c.bucket,
                        "intraday_return": round(c.intraday_return, 4),
                        "proximity_to_high": round(c.proximity_to_high, 4),
                        "volume_vs_avg": round(c.volume_vs_avg, 2),
                        "vs_market": round(c.vs_market, 4),
                        "atr_percent": round(c.atr_percent, 4),
                        "signal_price": round(c.signal_price, 4),
                        "adv_dollars": round(c.adv_dollars, 0),
                    }
                top_20_head = sorted(self.scored_candidates, key=lambda c: c.head_score, reverse=True)[:20]
                top_20_tail = sorted(self.scored_candidates, key=lambda c: c.composite_score, reverse=True)[:20]
                save_candidates_audit({
                    "top_20_by_head_score": [_candidate_dict(c) for c in top_20_head],
                    "top_20_by_composite_score": [_candidate_dict(c) for c in top_20_tail],
                })
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
                "v2_classified": self.v2_classified,
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
        self.v2_classified = bot_state.get("v2_classified", False)
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
        bot = OvernightMomentumBot()
    except Exception:
        logging.critical("UNHANDLED EXCEPTION during bot initialisation", exc_info=True)
        raise
    bot.run()


if __name__ == "__main__":
    main()
