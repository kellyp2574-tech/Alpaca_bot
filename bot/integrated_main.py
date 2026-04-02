"""Gap Momentum Bot - Main Orchestrator

Daily Schedule (ET):
- 09:00: Pull full market snapshot from Massive, filter by price ($0.50-$5.00)
- 09:25: Compute gaps from Massive data, build candidate list (core 4%+, filler 3-4%)
- 09:30:00: Submit full-size market orders at open for all candidates
- 09:31:00: Reconcile market order fills; if OPEN_ENTRY_PCT=1.0, finalize immediately
- 09:31:30: Rescue passes for partial fills (only when OPEN_ENTRY_PCT<1.0), finalize positions
- Variable exit: VIX-conditioned exits (2:30 PM low VIX, 3:30 PM high VIX, or trailing stop)
- 3:30/3:45/3:58 PM: Broker-based failsafe flatten sweeps
"""
import logging
import os
import sys
import time
from datetime import datetime, time as dt_time, timedelta
from typing import List, Optional, Dict, Any, Tuple
import json

from bot import config
from bot.massive_client import MassiveClient
from bot.market_data import AlpacaDataClient
from bot.gap_calculator import GapCalculator, GapCandidate
from bot.position_manager import PositionManager
from bot.vix_fetcher import VIXFetcher
from bot.state_manager import StateManager

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


class GapMomentumBot:
    """Main bot orchestrator for gap momentum strategy"""

    def __init__(self):
        self.massive = MassiveClient()
        self.alpaca = AlpacaDataClient()
        self.gap_calc = GapCalculator()
        self.position_mgr = PositionManager()
        self.vix_fetcher = VIXFetcher()
        self.state_mgr = StateManager()

        self.universe: List[str] = []
        self.candidates: List[GapCandidate] = []
        self.core_candidates: List[GapCandidate] = []  # 4%+ gaps
        self.filler_candidates: List[GapCandidate] = []  # 3-4% gaps
        self.massive_snapshots: Dict[str, Any] = {}
        self.vix_level: float = 15.0

        # Stage flags
        self.stage_universe_done = False
        self.stage_candidates_done = False
        self.stage_entry_done = False
        self.stage_exit_done = False

        # Staged entry control (prevents duplicate market order submission, NOT same as stage_entry_done)
        self.entry_submission_locked = False

        # Failsafe flags for broker-based flatten sweeps before market close
        self.failsafe_330_done = False
        self.failsafe_345_done = False
        self.failsafe_358_done = False

        # Post-close flatten retry cap (prevents infinite loop if positions can't close)
        self.post_close_flatten_attempts = 0
        self.max_post_close_flatten_attempts = 3

        # Cooldown: prevent mismatch flatten from firing every poll cycle
        self.mismatch_flatten_done = False

        # Periodic broker reconciliation timer (symbol-level sync every 60s during exits)
        self._last_broker_reconcile_time: Optional[datetime] = None

        # Rate-limit exit checks to stay under 200 API calls/min
        self.last_exit_check_time: Optional[datetime] = None
        self.exit_check_interval_seconds = 30

        # Retry counters
        self.universe_retry_count = 0
        self.max_universe_retries = config.UNIVERSE_MAX_RETRIES

        # Timing - Market orders at 9:30:00, reconcile at 9:31:00, rescue passes after
        self.target_universe = dt_time(9, 0)
        self.target_candidates = dt_time(9, 25)
        self.target_entry = dt_time(9, 30, 0)  # Submit market orders at 9:30:00
        self.market_close = dt_time(16, 0)

    def run(self):
        """Main bot loop - runs once per trading day"""
        logger.info("=" * 60)
        logger.info("Gap Momentum Bot Starting")
        logger.info("=" * 60)

        # Load any existing state (for recovery)
        self._load_state()

        # Startup market-hours guard (AFTER state load so we check restored entry_plans)
        current_time = datetime.now().time()
        
        # After market close: do not run
        if current_time >= self.market_close:
            logger.error(f"Current time {current_time} is after market close ({self.market_close}) - exiting")
            return
        
        # After 9:30:00 but before 9:31:00 with no saved staged-entry state: disable new entries
        if dt_time(9, 30, 0) <= current_time < dt_time(9, 31, 0) and not self.position_mgr.entry_plans:
            logger.warning("Startup in entry dead zone (9:30:00-9:31:00) with no saved entry plans - disabling new entries for the day")
            self.stage_entry_done = True
        
        # After 9:35:00 with no positions and no saved entry plans: disable entries, only run exit/failsafe
        # (Phase 3 broker repair runs at 9:35, so don't give up on entries until after that)
        if current_time >= dt_time(9, 35, 0) and self.position_mgr.get_position_count() == 0 and not self.position_mgr.entry_plans:
            logger.warning("Startup after 9:35:00 with no positions and no entry plans - disabling entries, will only run exit/failsafe logic")
            self.stage_entry_done = True
            self.stage_exit_done = True  # Nothing to exit

        # Main event loop
        while True:
            now = datetime.now()
            current_time = now.time()

            # Step 1: Universe reduction via Massive
            if not self.stage_universe_done and current_time >= self.target_universe:
                if current_time < self.target_candidates:
                    self._step1_build_universe()
                else:
                    logger.warning("Skipped Step 1 - past candidate window")
                    self.stage_universe_done = True

            # Step 2: Gap calculation via Massive
            if not self.stage_candidates_done and current_time >= self.target_candidates:
                if current_time < self.target_entry:
                    self._step2_find_candidates()
                else:
                    logger.warning("Skipped Step 2 - past entry window")
                    self.stage_candidates_done = True

            # Step 3: Market entry at 9:30:00, poll at 9:31:00, broker repair at 9:35:00
            # Uses three-phase staged entry: submit → poll → broker repair sync
            if config.USE_STAGED_OPEN_ENTRY:
                self._step3_manage_staged_entry(current_time)
            elif not self.stage_entry_done and current_time >= self.target_entry:
                self._step3_enter_positions()

            # Hard broker-based failsafe exits, independent of local bot memory
            if self.stage_entry_done and not self.failsafe_330_done and current_time >= dt_time(15, 30):
                # Clear exit slicers first so they don't fight with the failsafe
                self.position_mgr.exit_slicers.clear()
                self._run_failsafe_flatten("3:30 PM failsafe")
                self.failsafe_330_done = True

            if self.stage_entry_done and not self.failsafe_345_done and current_time >= dt_time(15, 45):
                self._run_failsafe_flatten("3:45 PM failsafe")
                self.failsafe_345_done = True

            if self.stage_entry_done and not self.failsafe_358_done and current_time >= dt_time(15, 58):
                self._run_failsafe_flatten("3:58 PM failsafe")
                self.failsafe_358_done = True

            # Step 4: Manage exits (rate-limited to every 30s to stay under 200 API calls/min)
            if self.stage_entry_done and not self.stage_exit_done:
                if current_time < self.market_close:
                    # Throttle exit checks: only run every exit_check_interval_seconds
                    should_check = (
                        self.last_exit_check_time is None
                        or (now - self.last_exit_check_time).total_seconds() >= self.exit_check_interval_seconds
                    )
                    if should_check:
                        self._step4_manage_exits(current_time)
                        self.last_exit_check_time = now
                else:
                    # Post-close: retry flatten with a cap to prevent infinite loop
                    if self.post_close_flatten_attempts < self.max_post_close_flatten_attempts:
                        self.post_close_flatten_attempts += 1
                        logger.warning(
                            f"Market close reached - flatten attempt "
                            f"{self.post_close_flatten_attempts}/{self.max_post_close_flatten_attempts}"
                        )
                        self._run_failsafe_flatten("4:00 PM market-close failsafe")
                    elif not self.stage_exit_done:
                        broker_count = self.position_mgr.broker_position_count()
                        if broker_count > 0:
                            logger.critical(
                                f"FAILED TO FLATTEN after {self.max_post_close_flatten_attempts} attempts - "
                                f"{broker_count} broker positions remain. Manual intervention required."
                            )
                        self.stage_exit_done = True

            # Check if day is complete
            if current_time >= self.market_close and self.stage_entry_done and self.stage_exit_done:
                logger.info("Market closed - day complete")
                self._finalize_day()
                break

            time.sleep(1)

    def _step1_build_universe(self):
        """Step 1: Pull full market snapshot from Massive, filter by price range"""
        logger.info("STEP 1: Building universe from Massive")

        try:
            snapshots = self.massive.get_full_market_snapshot()
            if not snapshots:
                self.universe_retry_count += 1
                if self.universe_retry_count >= self.max_universe_retries:
                    logger.error(f"Failed to get Massive snapshot after {self.max_universe_retries} retries")
                    self._fallback_to_alpaca_universe()
                else:
                    logger.warning(f"Empty Massive snapshot, retry {self.universe_retry_count}/{self.max_universe_retries}")
                    time.sleep(5)
                return

            self.universe = self.massive.filter_by_price_range(
                snapshots, config.MIN_PRICE, config.MAX_PRICE
            )
            
            # First-pass filter: remove symbols Alpaca marks as non-tradable (OTC, halted, etc.)
            # Uses bulk GET /v2/assets (1 API call). NOTE: this is not a complete shield —
            # symbols can still become close-only or restricted intraday due to broker
            # controls, corporate actions, or special restrictions.
            pre_filter_count = len(self.universe)
            tradable_set = set(self.alpaca.get_tradable_assets())
            if tradable_set:
                rejected = [s for s in self.universe if s not in tradable_set]
                self.universe = [s for s in self.universe if s in tradable_set]
                if rejected:
                    logger.info(f"Tradability filter: {pre_filter_count} -> {len(self.universe)} "
                               f"({len(rejected)} non-tradable removed)")
            else:
                logger.warning("Tradability filter skipped: failed to fetch Alpaca asset list")

            # Store full Massive snapshots for merging in Step 2
            self.massive_snapshots = snapshots

            logger.info(f"Universe built: {len(self.universe)} symbols")
            self.stage_universe_done = True
            self._save_state()

        except Exception as e:
            logger.error(f"Error in Step 1: {e}")
            self.universe_retry_count += 1
            if self.universe_retry_count >= self.max_universe_retries:
                self._fallback_to_alpaca_universe()

    def _fallback_to_alpaca_universe(self):
        """Fallback: Build universe from Alpaca assets when Massive fails"""
        logger.info("FALLBACK: Building universe from Alpaca assets")
        
        try:
            assets = self.alpaca.get_tradable_assets()
            if not assets:
                logger.error("Alpaca fallback also failed - no universe available")
                self.stage_universe_done = True
                return
            
            chunk_size = 1000
            min_target_universe = 500
            
            self.universe = []
            snapshots_received = False
            
            for i in range(0, len(assets), chunk_size):
                chunk = assets[i:i + chunk_size]
                snapshots = self.alpaca.get_snapshots(chunk)
                
                if snapshots:
                    snapshots_received = True
                
                for symbol, data in snapshots.items():
                    price = (
                        data.get("last_price")
                        or data.get("close")
                        or data.get("prev_close")
                        or 0
                    )
                    if price and config.MIN_PRICE <= price <= config.MAX_PRICE:
                        self.universe.append(symbol)
                
                if len(self.universe) >= min_target_universe:
                    break
            
            if snapshots_received:
                logger.info(f"Alpaca fallback universe: {len(self.universe)} symbols")
                self.stage_universe_done = True
                self._save_state()
            else:
                logger.error("Alpaca fallback: failed to get any snapshots")
                self.stage_universe_done = True
            
        except Exception as e:
            logger.error(f"Alpaca fallback error: {e}")
            self.stage_universe_done = True

    def _step2_find_candidates(self):
        """Step 2: Compute gaps from Massive data (primary), with Alpaca fallback for universe if needed"""
        logger.info("STEP 2: Finding gap candidates via Massive (Alpaca may be used for universe fallback in Step 1)")

        if not self.universe:
            logger.error("No universe available for Step 2")
            time.sleep(5)
            return

        try:
            # CRITICAL: Refresh Massive snapshot at 9:25 (don't use stale 9:00 data)
            logger.info("Refreshing Massive snapshot for candidate selection...")
            fresh_snapshots = self.massive.get_full_market_snapshot()
            if not fresh_snapshots:
                logger.error("Failed to refresh Massive snapshot - will retry")
                time.sleep(5)
                return

            # Filter to universe only (critical fix - was using full snapshot before)
            filtered_snapshots = {
                symbol: fresh_snapshots[symbol]
                for symbol in self.universe
                if symbol in fresh_snapshots
            }
            logger.info(f"Filtered to {len(filtered_snapshots)} universe symbols from Massive")

            # Store for pre-trade state save
            self.massive_snapshots = filtered_snapshots

            self.candidates = self.gap_calc.find_candidates(filtered_snapshots)

            # Split candidates: core (4%+) vs filler (3-4%)
            all_candidates = self.candidates
            core_candidates = [c for c in all_candidates if c.gap_pct >= 4.0]
            filler_candidates = [c for c in all_candidates if 3.0 <= c.gap_pct < 4.0]

            # Apply liquidity filter: core gets priority, filler gets remaining slots
            core_candidates = self.gap_calc.select_by_liquidity_and_gap(
                core_candidates, max_positions=config.MAX_POSITIONS
            )
            remaining_slots = max(0, config.MAX_POSITIONS - len(core_candidates))
            filler_candidates = self.gap_calc.select_by_liquidity_and_gap(
                filler_candidates, max_positions=remaining_slots
            ) if remaining_slots > 0 else []

            # Store both lists
            self.core_candidates = core_candidates
            self.filler_candidates = filler_candidates
            # Combined for backwards compatibility
            self.candidates = core_candidates + filler_candidates

            self.vix_level = self.vix_fetcher.get_vix_level() or 15.0

            logger.info(f"Candidates found: {len(self.candidates)} (core: {len(self.core_candidates)}, filler: {len(self.filler_candidates)})")
            logger.info(f"Core candidates (4%+):")
            for c in self.core_candidates[:5]:
                logger.info(f"  {c.symbol}: {c.gap_pct:+.1f}% gap, ${c.adv_estimate/1e6:.1f}M ADV")

            self.stage_candidates_done = True
            self._save_state()

        except Exception as e:
            logger.error(f"Error in Step 2: {e}")
            time.sleep(5)

    def _step3_enter_positions(self):
        """Step 3: DEPRECATED - Use _step3_manage_staged_entry for market-at-open execution.
        
        This legacy method sent partial OPG slices before 9:28 cutoff. The current design
        sends full market DAY orders at 9:30:00 via submit_open_entry_orders().
        """
        logger.warning("STEP 3: _step3_enter_positions is deprecated. Use staged entry flow.")
        self.stage_entry_done = True
        self._save_state()

    def _step3_manage_staged_entry(self, current_time: dt_time):
        """Three-stage entry: submit at 9:30, poll at 9:31, broker repair at 9:35.
        
        Phase 1 (9:30:00): Build plans + submit market orders.
        Phase 2 (9:31:00): Poll order status for 90s, finalize only confirmed fills.
                           Non-terminal orders stay pending.
        Phase 3 (9:35:00): Broker repair sync — fetch broker positions as ground truth,
                           resolve any still-pending orders, force-finalize all.
        """
        try:
            t_poll = dt_time(9, 31, 0)    # Start active polling
            t_repair = dt_time(9, 35, 0)  # Broker repair sync

            # ── Phase 1: Submit market orders at 9:30:00 ──
            if current_time >= self.target_entry and current_time < t_poll and not self.position_mgr.entry_stage1_done:
                logger.info("STAGED ENTRY PHASE 1: Build plans + submit market orders at open")

                if self.entry_submission_locked:
                    has_live_orders = any(
                        plan.open_order_id is not None
                        for plan in self.position_mgr.entry_plans.values()
                    )
                    if has_live_orders:
                        logger.warning(
                            "Crash recovery: lock set with persisted market orders but "
                            "entry_stage1_done=False — auto-promoting to stage1_done"
                        )
                        self.position_mgr.entry_stage1_done = True
                        self._save_state()
                    else:
                        logger.warning(
                            "Crash recovery: lock set but no market orders persisted — "
                            "unlocking for retry"
                        )
                        self.entry_submission_locked = False
                        self._save_state()
                    return

                self.entry_submission_locked = True
                self._save_state()

                try:
                    total_capital = self.position_mgr.get_total_capital()

                    self.position_mgr.entry_plans.clear()
                    plans_core = self.position_mgr.build_entry_plans(self.core_candidates, capital_override=total_capital)
                    
                    core_reserved_notional = sum(
                        plan.target_qty * plan.expected_open_price * 1.02
                        for plan in plans_core.values()
                    )
                    deployment_ratio = core_reserved_notional / total_capital if total_capital > 0 else 0
                    remaining_capital = total_capital - core_reserved_notional
                    remaining_slots = max(0, config.MAX_POSITIONS - len(plans_core))
                    
                    logger.info(f"PHASE 1: Built {len(plans_core)} core plans, "
                               f"reserved ${core_reserved_notional:,.2f} ({deployment_ratio*100:.1f}%), "
                               f"remaining ${remaining_capital:,.2f}, {remaining_slots} slots")
                    
                    plans_filler = {}
                    MIN_FILL_CAPITAL = 1000
                    
                    if (deployment_ratio < 0.8 and 
                        remaining_capital > MIN_FILL_CAPITAL and 
                        self.filler_candidates and 
                        remaining_slots > 0):
                        
                        filler_to_plan = self.filler_candidates[:remaining_slots]
                        plans_filler = self.position_mgr.build_entry_plans(
                            filler_to_plan, 
                            capital_override=remaining_capital
                        )
                        logger.info(f"Built {len(plans_filler)} filler plans against remaining capital")
                    else:
                        if deployment_ratio >= 0.8:
                            logger.info(f"Filler skipped: deployment {deployment_ratio*100:.1f}% >= 80%")
                        elif remaining_capital <= MIN_FILL_CAPITAL:
                            logger.info(f"Filler skipped: remaining ${remaining_capital:,.2f} <= min ${MIN_FILL_CAPITAL}")
                        elif remaining_slots <= 0:
                            logger.info(f"Filler skipped: no slots (core filled {len(plans_core)}/{config.MAX_POSITIONS})")
                        else:
                            logger.info(f"Filler skipped: no filler candidates")
                    
                    all_plans = {**plans_core, **plans_filler}
                    
                    self.position_mgr.submit_open_entry_orders(all_plans, state_saver=self._save_state)

                    logger.info(f"PHASE 1 COMPLETE: {len(plans_core)} core + {len(plans_filler)} filler = {len(all_plans)} orders submitted")

                    self.position_mgr.entry_stage1_done = True
                    self._save_state()
                    
                except Exception as e:
                    logger.exception(f"Error in staged entry phase 1: {e}")
                    
                    orders_submitted = any(
                        plan.open_order_id is not None 
                        for plan in self.position_mgr.entry_plans.values()
                    )
                    
                    if not orders_submitted:
                        logger.warning("No market orders were submitted - rolling back entry_submission_locked")
                        self.entry_submission_locked = False
                        self.position_mgr.entry_plans.clear()
                    else:
                        logger.warning(f"Partial submission: {sum(1 for p in self.position_mgr.entry_plans.values() if p.open_order_id)} orders submitted - keeping lock")
                    
                    self._save_state()
                
                return

            # ── Phase 2: Active poll + partial finalize at 9:31:00 ──
            # Polls for 90s. Only finalizes orders confirmed terminal.
            # Non-terminal orders stay pending for Phase 3.
            if current_time >= t_poll and current_time < t_repair and self.position_mgr.entry_stage1_done and not self.position_mgr.entry_stage2_done:
                logger.info("PHASE 2: Active poll of market order fills (90s window)")
                self.position_mgr.reconcile_open_order_fills()

                # Finalize only confirmed fills (terminal orders with qty > 0)
                confirmed = self.position_mgr.finalize_entry_positions(force=False)
                if confirmed:
                    logger.info(f"Phase 2: finalized {len(confirmed)} confirmed fills")
                    for pos in confirmed:
                        logger.info(f"  {pos.symbol}: {pos.quantity} @ ${pos.entry_price:.4f}")

                # Count pending (non-terminal) orders
                pending = sum(
                    1 for plan in self.position_mgr.entry_plans.values()
                    if plan.open_order_id and not plan.open_order_terminal and not plan.finalized
                )
                if pending > 0:
                    logger.warning(f"Phase 2: {pending} orders still pending — will resolve at 9:35 broker repair")
                else:
                    logger.info("Phase 2: all orders terminal — entry complete")
                    self.stage_entry_done = True

                self.position_mgr.entry_stage2_done = True
                self._save_state()
                return

            # ── Phase 3: Broker repair sync at 9:35:00, retries until 9:38 hard cutoff ──
            # Resolves pending orders using broker positions as ground truth.
            # Re-enters on each main loop iteration until all orders are resolved or 9:38 hits.
            t_hard_cutoff = dt_time(9, 38, 0)

            if current_time >= t_repair and self.position_mgr.entry_stage2_done and not self.stage_entry_done:
                past_cutoff = current_time >= t_hard_cutoff

                if past_cutoff:
                    logger.info("PHASE 3: Broker repair sync — HARD CUTOFF 9:38, force-finalizing all")
                else:
                    logger.info("PHASE 3: Broker repair sync — resolving pending orders")

                # If OPEN_ENTRY_PCT < 1.0, run rescue passes on first Phase 3 entry only
                if config.OPEN_ENTRY_PCT < 1.0 and not hasattr(self, '_rescue_passes_done'):
                    logger.info("Running rescue passes for remaining size")
                    self.position_mgr.refresh_no_fill_prices()
                    self.position_mgr.submit_post_open_rescue_pass("market1", state_saver=self._save_state)
                    self.position_mgr.submit_post_open_rescue_pass("market2", state_saver=self._save_state)
                    self._rescue_passes_done = True

                # Broker repair: resolve any still-pending orders
                repaired = self.position_mgr.broker_repair_sync()

                # Check if any orders are still unresolved
                still_pending = sum(
                    1 for plan in self.position_mgr.entry_plans.values()
                    if plan.open_order_id and not plan.open_order_terminal and not plan.finalized
                )

                if still_pending > 0 and not past_cutoff:
                    # Orders still unresolved — retry on next loop iteration
                    # Finalize only what's confirmed so far
                    confirmed = self.position_mgr.finalize_entry_positions(force=False)
                    if confirmed:
                        logger.info(f"Phase 3: finalized {len(confirmed)} newly confirmed fills")
                    logger.warning(
                        f"Phase 3: {still_pending} orders still unresolved — "
                        f"will retry next cycle (hard cutoff at 9:38)"
                    )
                    self._save_state()
                    return

                # All resolved or past hard cutoff — force-finalize everything
                positions = self.position_mgr.finalize_entry_positions(force=True)
                total_positions = self.position_mgr.get_position_count()

                logger.info(
                    f"PHASE 3 COMPLETE: {len(repaired)} repaired from broker, "
                    f"{len(positions)} new positions finalized, "
                    f"{total_positions} total positions"
                )
                for pos in positions:
                    logger.info(f"  {pos.symbol}: {pos.quantity} @ ${pos.entry_price:.4f}")

                self.stage_entry_done = True
                self._save_state()
                return
        except Exception as e:
            logger.exception(f"Error in staged entry flow: {e}")
            self._save_state()

    def _step4_manage_exits(self, current_time: dt_time):
        """Step 4: Monitor and execute exits based on VIX regime.
        
        Includes periodic symbol-level broker reconciliation (every 60s) to catch:
        - Late fills that slipped past the Phase 3 hard cutoff
        - Symbol mismatches (local and broker have same count but different names)
        - Wrong local quantities from partial fill desync
        """
        local_count = self.position_mgr.get_position_count()
        broker_count = self.position_mgr.broker_position_count()

        if local_count == 0 and broker_count == 0:
            logger.info("All positions closed locally and at broker - exit complete")
            self.stage_exit_done = True
            self._save_state()
            return

        # ── Periodic symbol-level broker reconciliation (every 60s) ──
        # This catches late fills, symbol mismatches, and wrong quantities.
        # Runs after the entry grace period (9:40+) to avoid conflicting with Phase 3.
        now = datetime.now()
        if current_time >= dt_time(9, 40):
            if (self._last_broker_reconcile_time is None or
                    (now - self._last_broker_reconcile_time).total_seconds() >= 60):
                actions = self.position_mgr.reconcile_local_positions_from_broker()
                self._last_broker_reconcile_time = now
                if actions:
                    self._save_state()
                # Re-read counts after reconciliation
                local_count = self.position_mgr.get_position_count()
                broker_count = self.position_mgr.broker_position_count()

        if local_count == 0 and broker_count == 0:
            logger.info("All positions closed locally and at broker - exit complete")
            self.stage_exit_done = True
            self._save_state()
            return

        if local_count == 0 and broker_count > 0:
            # Guard: don't flatten during entry grace period (before 9:40).
            if current_time < dt_time(9, 40):
                logger.warning(
                    f"Local empty but broker shows {broker_count} positions — "
                    f"in entry grace period (before 9:40), NOT flattening"
                )
                return

            if not self.mismatch_flatten_done:
                logger.warning(
                    f"Local positions empty but broker shows {broker_count} live positions - "
                    f"running immediate broker flatten"
                )
                self._run_failsafe_flatten("intraday broker/local mismatch")
                self.mismatch_flatten_done = True
                self._save_state()
            else:
                logger.warning(
                    f"Broker still shows {broker_count} positions after mismatch flatten - "
                    f"waiting for failsafe sweeps"
                )
            return

        if local_count > 0 and broker_count == 0:
            logger.warning(
                f"Local shows {local_count} positions but broker is flat - "
                f"clearing stale local state"
            )
            self.position_mgr.positions.clear()
            self.position_mgr.exit_slicers.clear()
            self.stage_exit_done = True
            self._save_state()
            return

        current_prices = self.position_mgr.update_positions()
        exited = self.position_mgr.check_exits(current_time, self.vix_level, current_prices)
        if exited:
            logger.info(f"Exited {len(exited)} positions: {exited}")
            self._save_state()

    def _load_state(self):
        """Load state from previous run (for same-day crash recovery only).
        
        Rules:
        1. Non-today state → nuke everything, start completely fresh.
        2. Same-day state → restore only if backing data is complete.
        3. Broker positions are always ground truth at the end.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        bot_state = self.state_mgr.load_bot_state()
        saved_date = bot_state.get("date") if bot_state else None
        is_same_day = saved_date == today

        # ── Step 1: If not same-day, clear everything ──
        if not is_same_day:
            if saved_date:
                logger.warning(f"Stale state from {saved_date} — clearing all state for fresh start")
            else:
                logger.info("No saved state found — fresh start")
            self.state_mgr.clear_bot_state()
            self.state_mgr.save_positions({})
            self._clear_pre_trade_state()
            self.position_mgr.positions.clear()
            self.position_mgr.entry_plans.clear()
            self.position_mgr.exit_slicers.clear()
            # All stage flags stay at __init__ defaults (False)

        # ── Step 2: Same-day recovery ──
        else:
            logger.info(f"Same-day state found — attempting recovery")

            # Restore entry plans if saved
            entry_plans_data = bot_state.get("entry_plans", {})
            if entry_plans_data:
                from bot.position_manager import EntryExecutionPlan
                for symbol, plan_data in entry_plans_data.items():
                    self.position_mgr.entry_plans[symbol] = EntryExecutionPlan(**plan_data)
                logger.info(f"Restored {len(entry_plans_data)} entry plans")

            # Restore positions from file
            positions = self.state_mgr.load_positions()
            if positions:
                self.position_mgr.load_positions(positions)
                logger.info(f"Restored {len(positions)} positions from state")

            # Restore pre-trade data (universe, candidates, Massive snapshots)
            pre_trade = self._load_pre_trade_state()
            if pre_trade:
                self.universe = pre_trade.get("universe", [])
                self.massive_snapshots = pre_trade.get("massive_snapshots", {})
                candidates_data = pre_trade.get("candidates", [])
                if candidates_data:
                    self.candidates = [GapCandidate(**c) for c in candidates_data]
                    self.core_candidates = [c for c in self.candidates if c.gap_pct >= 4.0]
                    self.filler_candidates = [c for c in self.candidates if 3.0 <= c.gap_pct < 4.0]
                    logger.info(f"Restored {len(self.candidates)} candidates: {len(self.core_candidates)} core, {len(self.filler_candidates)} filler")

            # Restore stage flags ONLY if we have the backing data to support them
            has_universe = bool(self.universe)
            has_candidates = bool(self.candidates)
            has_entry_plans = bool(self.position_mgr.entry_plans)
            has_positions = self.position_mgr.get_position_count() > 0

            if has_universe:
                self.stage_universe_done = bot_state.get("stage_universe_done", False)
            if has_candidates:
                self.stage_candidates_done = bot_state.get("stage_candidates_done", False)
            if has_entry_plans or has_positions:
                self.stage_entry_done = bot_state.get("stage_entry_done", False)
                self.entry_submission_locked = bot_state.get("entry_submission_locked", False)
                self.position_mgr.entry_stage1_done = bot_state.get("entry_stage1_done", False)
                self.position_mgr.entry_stage2_done = bot_state.get("entry_stage2_done", False)
            if has_positions:
                self.stage_exit_done = bot_state.get("stage_exit_done", False)

            self.vix_level = bot_state.get("vix_level", self.vix_level)

            logger.info(
                f"Recovery: universe={len(self.universe)}, candidates={len(self.candidates)}, "
                f"plans={len(self.position_mgr.entry_plans)}, positions={self.position_mgr.get_position_count()}, "
                f"stages: U={self.stage_universe_done} C={self.stage_candidates_done} "
                f"E={self.stage_entry_done} X={self.stage_exit_done}"
            )

        # ── Step 3: Cancel orphaned orders ──
        if self.position_mgr.entry_plans:
            canceled = self.position_mgr.cancel_orphaned_open_orders()
            if canceled > 0:
                logger.warning(f"Startup: Canceled {canceled} orphaned open orders (not matching restored entry_plans)")
        else:
            canceled = self.position_mgr.cancel_open_buy_orders()
            if canceled > 0:
                logger.warning(f"Startup: Canceled {canceled} orphaned open buy orders")

        # ── Step 4: Broker is ground truth — reconcile ──
        live_broker_positions = self.position_mgr.get_broker_positions()
        local_count = self.position_mgr.get_position_count()

        if live_broker_positions:
            logger.warning(f"Startup: broker shows {len(live_broker_positions)} live positions")
            for pos in live_broker_positions:
                logger.warning(f"  Broker: {pos.get('symbol')} qty={pos.get('qty')} side={pos.get('side')}")
            self.position_mgr.reconcile_local_positions_from_broker()
            # If broker has positions, entries are done.
            # NOTE: reconcile_local_positions_from_broker rebuilds positions with
            # zeroed entry metadata (gap_pct=0, adv_estimate=0). Exit logic that
            # depends on original entry context (e.g., gap-based trailing stops)
            # will lose fidelity. This is an acceptable tradeoff vs losing positions.
            self.stage_entry_done = True
            self.stage_universe_done = True
            self.stage_candidates_done = True
        elif local_count > 0:
            logger.warning(f"Local shows {local_count} positions but broker is flat — clearing stale local state")
            self.position_mgr.positions.clear()
            self.position_mgr.exit_slicers.clear()

    def _save_state(self):
        """Save current state including positions and bot state"""
        self.state_mgr.save_positions(self.position_mgr.positions)
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Serialize entry plans for persistence
        entry_plans_data = {}
        for symbol, plan in self.position_mgr.entry_plans.items():
            entry_plans_data[symbol] = {
                "symbol": plan.symbol,
                "target_qty": plan.target_qty,
                "open_qty": plan.open_qty,
                "planned_remaining_qty": plan.planned_remaining_qty,
                "expected_open_price": plan.expected_open_price,
                "gap_pct": plan.gap_pct,
                "adv_estimate": plan.adv_estimate,
                "open_order_id": plan.open_order_id,
                "open_filled_qty": plan.open_filled_qty,
                "open_filled_avg_price": plan.open_filled_avg_price,
                "market1_order_id": plan.market1_order_id,
                "market1_filled_qty": plan.market1_filled_qty,
                "market1_filled_avg_price": plan.market1_filled_avg_price,
                "market2_order_id": plan.market2_order_id,
                "market2_filled_qty": plan.market2_filled_qty,
                "market2_filled_avg_price": plan.market2_filled_avg_price,
                "finalized": plan.finalized,
                "open_order_terminal": plan.open_order_terminal,
            }
        
        self.state_mgr.save_bot_state({
            "date": today,
            "vix_level": self.vix_level,
            "stage_universe_done": self.stage_universe_done,
            "stage_candidates_done": self.stage_candidates_done,
            "stage_entry_done": self.stage_entry_done,
            "stage_exit_done": self.stage_exit_done,
            # Staged entry state
            "entry_submission_locked": self.entry_submission_locked,
            "entry_stage1_done": self.position_mgr.entry_stage1_done,
            "entry_stage2_done": self.position_mgr.entry_stage2_done,
            "entry_plans": entry_plans_data,
        })
        
        # CRITICAL FIX: Also save pre-trade state (universe, candidates, Massive snapshots)
        self._save_pre_trade_state()

    def _save_pre_trade_state(self):
        """Save universe, candidates, and Massive snapshots for recovery with date"""
        pre_trade_file = os.path.join(config.STATE_DIR, "pre_trade_state.json")
        today = datetime.now().strftime("%Y-%m-%d")
        state = {
            "date": today,
            "universe": self.universe,
            "massive_snapshots": self.massive_snapshots,
        }
        if self.candidates:
            state["candidates"] = [
                {
                    "symbol": c.symbol,
                    "open_price": c.open_price,
                    "prev_close": c.prev_close,
                    "gap_pct": c.gap_pct,
                    "volume": c.volume,
                    "adv_estimate": c.adv_estimate,
                }
                for c in self.candidates
            ]
        try:
            with open(pre_trade_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving pre-trade state: {e}")

    def _load_pre_trade_state(self) -> Optional[dict]:
        """Load universe and Massive data for recovery with date validation"""
        pre_trade_file = os.path.join(config.STATE_DIR, "pre_trade_state.json")
        if not os.path.exists(pre_trade_file):
            return None
        try:
            with open(pre_trade_file, 'r') as f:
                state = json.load(f)
            
            # Validate date - don't use stale pre-trade state
            today = datetime.now().strftime("%Y-%m-%d")
            state_date = state.get("date")
            if state_date != today:
                logger.warning(f"Pre-trade state from {state_date} is stale (today: {today}) - ignoring")
                return None
            
            return state
        except Exception as e:
            logger.error(f"Error loading pre-trade state: {e}")
            return None

    def _clear_pre_trade_state(self):
        """Clear pre-trade state file"""
        pre_trade_file = os.path.join(config.STATE_DIR, "pre_trade_state.json")
        if os.path.exists(pre_trade_file):
            os.remove(pre_trade_file)

    def _run_failsafe_flatten(self, label: str):
        """Broker-based catch-all flatten."""
        logger.warning(f"{label}: starting broker-based failsafe flatten")

        summary = self.position_mgr.force_flatten_broker_positions(label)

        logger.warning(
            f"{label}: failsafe flatten complete | "
            f"positions_seen={summary['positions_seen']} | "
            f"closes_submitted={summary['closes_submitted']} | "
            f"fills_confirmed={summary['fills_confirmed']} | "
            f"errors={len(summary['errors'])}"
        )

        # If broker is flat, mark exits complete and clear any stale local state
        if self.position_mgr.broker_position_count() == 0:
            self.stage_exit_done = True
            self.position_mgr.positions.clear()
            self.position_mgr.exit_slicers.clear()
            logger.warning(f"{label}: broker confirmed flat — local state cleared")
        else:
            logger.error(f"{label}: broker still shows open positions after failsafe")

        self._save_state()

    def _finalize_day(self):
        """Finalize trading day, log summary"""
        logger.info("Finalizing trading day")

        broker_count_before = self.position_mgr.broker_position_count()
        if broker_count_before > 0:
            logger.warning(f"Finalize day: broker shows {broker_count_before} open positions, flattening")
            self.position_mgr.force_flatten_broker_positions("finalize_day")

        broker_count_after = self.position_mgr.broker_position_count()
        if broker_count_after > 0:
            logger.critical(
                f"FAILED TO FLATTEN BROKER POSITIONS AT END OF DAY - "
                f"{broker_count_after} still open, preserving state"
            )
            return

        self.state_mgr.clear_positions()
        self.state_mgr.clear_bot_state()
        self._clear_pre_trade_state()

        # CRITICAL: Explicitly clear staged entry state to prevent leakage to next day
        self.position_mgr.entry_plans.clear()
        self.position_mgr.entry_stage1_done = False
        self.position_mgr.entry_stage2_done = False
        self.entry_submission_locked = False
        logger.info("Cleared staged entry state (entry_plans, stage flags, lock)")

        today = datetime.now().strftime("%Y-%m-%d")
        summary = {
            "universe_size": len(self.universe),
            "candidates_found": len(self.candidates),
            "vix_level": self.vix_level,
            "stage_entry_done": self.stage_entry_done,
            "stage_exit_done": self.stage_exit_done,
        }
        self.state_mgr.log_daily_summary(today, summary)

        logger.info("Day complete - Goodbye!")


def main():
    bot = GapMomentumBot()
    try:
        bot.run()
    except KeyboardInterrupt:
        current_time = datetime.now().time()
        logger.warning(f"Interrupted by user at {current_time}")
        
        # Only flatten if after market close or in failsafe window
        if current_time >= dt_time(16, 0):
            logger.info("After market close - running finalize_day")
            bot._finalize_day()
        elif current_time >= dt_time(15, 30):
            logger.warning("In failsafe window - running broker flatten only")
            bot._run_failsafe_flatten("KeyboardInterrupt during failsafe window")
            bot._save_state()
        else:
            logger.warning("During market hours - preserving state, NOT flattening positions")
            logger.warning("Positions remain open. Manual intervention required if liquidation needed.")
            bot._save_state()
    except Exception as e:
        current_time = datetime.now().time()
        logger.exception(f"Fatal error at {current_time}: {e}")
        
        # Only flatten if after market close or in failsafe window
        if current_time >= dt_time(16, 0):
            logger.info("After market close - running finalize_day")
            bot._finalize_day()
        elif current_time >= dt_time(15, 30):
            logger.warning("In failsafe window - running broker flatten only")
            bot._run_failsafe_flatten("Exception during failsafe window")
            bot._save_state()
        else:
            logger.critical("FATAL ERROR DURING MARKET HOURS - preserving state, NOT flattening positions")
            logger.critical("Positions remain open. Manual intervention required.")
            bot._save_state()
        raise


if __name__ == "__main__":
    main()
