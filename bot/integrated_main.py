"""Gap Momentum Bot - Main Orchestrator

Daily Schedule (ET):
- 09:00: Pull full market snapshot from Massive, filter by price ($1-$2)
- 09:25: Compute gaps from Massive data, build candidate list
- 09:27: Submit MOO (Market On Open) orders for qualifying candidates
- Variable exit: VIX-conditioned exits (2:30 PM, 3:30 PM, or trailing stop)
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

        # Retry counters
        self.universe_retry_count = 0
        self.max_universe_retries = 3

        # Timing - MOO orders at 9:27 (cutoff ~9:28), candidates at 9:25
        self.target_universe = dt_time(9, 0)
        self.target_candidates = dt_time(9, 25)
        self.target_entry = dt_time(9, 27)
        self.market_close = dt_time(16, 0)

    def run(self):
        """Main bot loop - runs once per trading day"""
        logger.info("=" * 60)
        logger.info("Gap Momentum Bot Starting")
        logger.info("=" * 60)

        # Load any existing state (for recovery)
        self._load_state()

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

            # Step 2: Gap calculation via Alpaca
            if not self.stage_candidates_done and current_time >= self.target_candidates:
                if current_time < self.target_entry:
                    self._step2_find_candidates()
                else:
                    logger.warning("Skipped Step 2 - past entry window")
                    self.stage_candidates_done = True

            # Step 3: Market entry (MOO orders must submit before 9:28 cutoff)
            # Use staged entry if enabled: MOO slice before open, rescue passes after
            if config.USE_STAGED_OPEN_ENTRY and not self.stage_entry_done:
                self._step3_manage_staged_entry(current_time)
            elif not self.stage_entry_done and current_time >= self.target_entry:
                self._step3_enter_positions()

            # Step 4: Manage exits
            if self.stage_entry_done and not self.stage_exit_done:
                if current_time < self.market_close:
                    self._step4_manage_exits(current_time)
                else:
                    logger.info("Market close reached - forcing exits before finalizing")
                    if self.position_mgr.get_position_count() > 0:
                        self.position_mgr.force_exit_all("day_end")
                    
                    if self.position_mgr.get_position_count() == 0:
                        self.stage_exit_done = True
                    else:
                        logger.error("Market close exit attempted, but positions remain open")

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
        """Step 2: Compute gaps from Massive data only, find candidates"""
        logger.info("STEP 2: Finding gap candidates via Massive (no Alpaca dependency)")

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

            # Apply liquidity filter separately
            core_candidates = self.gap_calc.select_by_liquidity_and_gap(
                core_candidates, max_positions=config.MAX_POSITIONS
            )
            filler_candidates = self.gap_calc.select_by_liquidity_and_gap(
                filler_candidates, max_positions=config.MAX_POSITIONS
            )

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
        """Step 3: Submit MOO orders for candidates (must complete before 9:28 cutoff)"""
        logger.info("STEP 3: Entering positions via MOO orders")

        # HARD CUTOFF: Alpaca OPG cutoff is ~9:28, use 9:27:30 for 30s safety buffer
        current_time = datetime.now().time()
        if current_time >= dt_time(9, 27, 30):
            logger.error(f"Past OPG cutoff (9:27:30 buffer); current time {current_time}. Skipping entry stage.")
            self.stage_entry_done = True
            self._save_state()
            return

        if not self.candidates:
            logger.info("No candidates to enter")
            self.stage_entry_done = True
            self._save_state()
            return

        # CRITICAL FIX: Lock stage immediately to prevent re-entry race condition
        if self.stage_entry_done:
            logger.info("Execution guard: stage_entry_done flag is True - skipping entry")
            return

        # Check for existing positions before locking
        if self.position_mgr.get_position_count() > 0:
            logger.warning(f"Execution guard: {self.position_mgr.get_position_count()} positions already exist - skipping entry")
            self.stage_entry_done = True
            self._save_state()
            return

        # LOCK IMMEDIATELY - prevents any re-entry
        self.stage_entry_done = True
        self._save_state()
        logger.info("Entry stage locked - stage_entry_done = True")

        try:
            # Fetch total capital ONCE before deployment for consistent calculations
            total_capital = self.position_mgr.get_total_capital()
            
            # PHASE 1: Deploy to core candidates (4%+ gaps)
            logger.info(f"PHASE 1: Deploying to {len(self.core_candidates)} core candidates (4%+ gaps) using ${total_capital:,.2f} total capital")
            positions_core, used_capital = self.position_mgr.enter_positions_moo(
                self.core_candidates, self.vix_level, capital_override=total_capital
            )
            
            # PHASE 2: Compute deployment metrics using the ORIGINAL total_capital
            deployment_ratio = used_capital / total_capital if total_capital > 0 else 0
            remaining_capital = total_capital - used_capital
            
            logger.info(f"Phase 1 complete: {len(positions_core)} positions, "
                       f"${used_capital:,.2f} used, "
                       f"{deployment_ratio*100:.1f}% deployed, "
                       f"${remaining_capital:,.2f} remaining")
            
            # PHASE 3: Conditionally unlock filler candidates (3-4% gaps)
            positions_filler = []
            MIN_FILL_CAPITAL = 1000  # Minimum capital needed to enter filler positions
            
            # Enforce global position cap: filler can only use remaining slots after core
            core_position_count = len(positions_core)
            remaining_slots = max(0, config.MAX_POSITIONS - core_position_count)
            
            if deployment_ratio < 0.8 and remaining_capital > MIN_FILL_CAPITAL and self.filler_candidates and remaining_slots > 0:
                # Limit filler candidates to remaining slots
                filler_to_use = self.filler_candidates[:remaining_slots]
                logger.info(f"PHASE 3: Unlocking filler candidates (3-4% gaps) - "
                           f"deployment {deployment_ratio*100:.1f}% < 80%, "
                           f"${remaining_capital:,.2f} available, "
                           f"{remaining_slots} slots remaining for filler")
                positions_filler, _ = self.position_mgr.enter_positions_moo(
                    filler_to_use,
                    self.vix_level,
                    capital_override=remaining_capital
                )
            else:
                if deployment_ratio >= 0.8:
                    logger.info(f"Phase 3 skipped: deployment {deployment_ratio*100:.1f}% >= 80%")
                elif remaining_capital <= MIN_FILL_CAPITAL:
                    logger.info(f"Phase 3 skipped: remaining capital ${remaining_capital:,.2f} <= minimum ${MIN_FILL_CAPITAL}")
                elif remaining_slots <= 0:
                    logger.info(f"Phase 3 skipped: no remaining slots (core filled {core_position_count}/{config.MAX_POSITIONS})")
                else:
                    logger.info(f"Phase 3 skipped: no filler candidates available")
            
            # Combine results
            positions = positions_core + positions_filler
            
            logger.info(f"Entered {len(positions)} total positions ({len(positions_core)} core, {len(positions_filler)} filler)")
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.entry_price:.2f}")

            self._save_state()

        except Exception as e:
            logger.error(f"Error in Step 3: {e}")
            # Do NOT unlock - safer to fail than risk duplicate orders
            # If manual retry is needed, user must clear state manually
            time.sleep(5)

    def _step3_manage_staged_entry(self, current_time: dt_time):
        """Three-stage entry: pre-open MOO, 9:30:10 rescue, 9:30:30 cleanup."""
        t1 = datetime.strptime(config.POST_OPEN_ENTRY_TIME_1, "%H:%M:%S").time()
        t2 = datetime.strptime(config.POST_OPEN_ENTRY_TIME_2, "%H:%M:%S").time()

        # Hard pre-open cutoff
        if current_time >= dt_time(9, 27, 30) and not self.position_mgr.entry_plans and not self.stage_entry_done:
            logger.warning("Past MOO cutoff before entry plans were built")
            self.stage_entry_done = True
            self._save_state()
            return

        # Pre-open MOO submission
        if current_time >= self.target_entry and current_time < t1 and not self.position_mgr.entry_stage1_done:
            logger.info("STAGED ENTRY PHASE 1: Build plans + submit MOO slice")

            if self.stage_entry_done:
                return

            self.stage_entry_done = True
            self._save_state()

            total_capital = self.position_mgr.get_total_capital()

            plans_core = self.position_mgr.build_entry_plans(self.core_candidates, capital_override=total_capital)
            self.position_mgr.submit_moo_entry_slice(plans_core)

            self.position_mgr.entry_stage1_done = True
            self._save_state()
            return

        # First rescue pass
        if current_time >= t1 and current_time < t2 and self.position_mgr.entry_stage1_done and not self.position_mgr.entry_stage2_done:
            logger.info("STAGED ENTRY PHASE 2: reconcile MOO + rescue pass 1")
            self.position_mgr.reconcile_moo_fills()
            self.position_mgr.submit_post_open_rescue_pass("market1")
            self.position_mgr.entry_stage2_done = True
            self._save_state()
            return

        # Final rescue + finalize
        if current_time >= t2 and self.position_mgr.entry_stage2_done:
            logger.info("STAGED ENTRY PHASE 3: rescue pass 2 + finalize")
            self.position_mgr.submit_post_open_rescue_pass("market2")
            positions = self.position_mgr.finalize_entry_positions()

            logger.info(f"Entered {len(positions)} staged positions")
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.quantity} @ ${pos.entry_price:.4f}")

            self._save_state()

    def _step4_manage_exits(self, current_time: dt_time):
        """Step 4: Monitor and execute exits based on VIX regime"""
        if self.position_mgr.get_position_count() == 0:
            logger.info("All positions closed - exit complete")
            self.stage_exit_done = True
            return

        current_prices = self.position_mgr.update_positions()
        exited = self.position_mgr.check_exits(current_time, self.vix_level, current_prices)
        if exited:
            logger.info(f"Exited {len(exited)} positions: {exited}")

        self._save_state()

    def _load_state(self):
        """Load state from previous run (for recovery)"""
        # Load positions first
        positions = self.state_mgr.load_positions()
        if positions:
            logger.info(f"Loaded {len(positions)} positions from state, restoring...")
            self.position_mgr.load_positions(positions)
            self.stage_entry_done = True
            self.stage_universe_done = True
            self.stage_candidates_done = True
        
        # Load bot state (VIX, stages, and pre-trade data)
        bot_state = self.state_mgr.load_bot_state()
        today = datetime.now().strftime("%Y-%m-%d")
        
        if bot_state:
            saved_date = bot_state.get("date")
            if saved_date == today:
                self.vix_level = bot_state.get("vix_level", self.vix_level)
                
                # CRITICAL FIX: Only restore stage flags if we also restore the underlying data
                # Check if we have pre-trade state saved
                pre_trade = self._load_pre_trade_state()
                if pre_trade:
                    self.universe = pre_trade.get("universe", [])
                    self.massive_snapshots = pre_trade.get("massive_snapshots", {})
                    candidates_data = pre_trade.get("candidates", [])
                    if candidates_data:
                        self.candidates = [GapCandidate(**c) for c in candidates_data]
                        # CRITICAL FIX: Rebuild core/filler split from restored candidates
                        self.core_candidates = [c for c in self.candidates if c.gap_pct >= 4.0]
                        self.filler_candidates = [c for c in self.candidates if 3.0 <= c.gap_pct < 4.0]
                        logger.info(f"Restored {len(self.candidates)} candidates: {len(self.core_candidates)} core, {len(self.filler_candidates)} filler")
                    
                    # Now safe to restore stage flags
                    self.stage_universe_done = bot_state.get("stage_universe_done", False)
                    self.stage_candidates_done = bot_state.get("stage_candidates_done", False)
                    self.stage_entry_done = bot_state.get("stage_entry_done", False)
                    self.stage_exit_done = bot_state.get("stage_exit_done", False)
                    
                    logger.info(f"Restored full state: universe={len(self.universe)}, candidates={len(self.candidates)}, stages: U={self.stage_universe_done}, C={self.stage_candidates_done}, E={self.stage_entry_done}, X={self.stage_exit_done}")
                else:
                    # No pre-trade data - don't restore stage flags for steps 1-2
                    logger.warning("No pre-trade state found - will rebuild universe and candidates")
                    # Only restore entry/exit if we have positions
                    if positions:
                        self.stage_entry_done = True
                        self.stage_universe_done = True
                        self.stage_candidates_done = True
            else:
                logger.warning(f"Stale bot state from {saved_date} - starting fresh")
                self.state_mgr.clear_bot_state()
                self._clear_pre_trade_state()

    def _save_state(self):
        """Save current state including positions and bot state"""
        self.state_mgr.save_positions(self.position_mgr.positions)
        
        today = datetime.now().strftime("%Y-%m-%d")
        self.state_mgr.save_bot_state({
            "date": today,
            "vix_level": self.vix_level,
            "stage_universe_done": self.stage_universe_done,
            "stage_candidates_done": self.stage_candidates_done,
            "stage_entry_done": self.stage_entry_done,
            "stage_exit_done": self.stage_exit_done,
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

    def _finalize_day(self):
        """Finalize trading day, log summary"""
        logger.info("Finalizing trading day")

        if self.position_mgr.get_position_count() > 0:
            logger.warning("Force exiting remaining positions")
            self.position_mgr.force_exit_all("day_end")
        
        if self.position_mgr.get_position_count() > 0:
            logger.critical("FAILED TO FLATTEN POSITIONS AT END OF DAY - preserving state")
            return

        self.state_mgr.clear_positions()
        self.state_mgr.clear_bot_state()
        self._clear_pre_trade_state()

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
        logger.info("Interrupted by user")
        bot._finalize_day()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        bot._finalize_day()
        raise


if __name__ == "__main__":
    main()
