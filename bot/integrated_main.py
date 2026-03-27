"""Gap Momentum Bot - Main Orchestrator

Daily Schedule (ET):
- 09:00: Pull full market snapshot from Massive, filter by price ($1-$2)
- 09:29: Fetch Alpaca IEX snapshots, compute gaps, build candidate list
- 09:30: Enter market orders for qualifying candidates
- Variable exit: VIX-conditioned exits (2:30 PM, 3:30 PM, or trailing stop)
"""
import logging
import os
import sys
import time
from datetime import datetime, time as dt_time, timedelta
from typing import List, Optional, Dict
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
        self.vix_level: float = 0.0
        self.massive_snapshots: Dict[str, dict] = {}

        # Stage flags
        self.stage_universe_done = False
        self.stage_candidates_done = False
        self.stage_entry_done = False
        self.stage_exit_done = False

        # Retry counters
        self.universe_retry_count = 0
        self.max_universe_retries = 3

        # Timing
        self.target_universe = dt_time(9, 0)
        self.target_candidates = dt_time(9, 29)
        self.target_entry = dt_time(9, 30)
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

            # Step 3: Market entry
            if not self.stage_entry_done and current_time >= self.target_entry:
                if current_time < dt_time(10, 0):
                    self._step3_enter_positions()
                else:
                    logger.warning("Skipped Step 3 - past entry retry window")
                    self.stage_entry_done = True

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
        """Step 2: Fetch Alpaca snapshots, merge Massive data, compute gaps, find candidates"""
        logger.info("STEP 2: Finding gap candidates via Alpaca")

        if not self.universe:
            logger.error("No universe available for Step 2")
            time.sleep(5)
            return

        try:
            snapshots = self.alpaca.get_snapshots(self.universe)
            if not snapshots:
                logger.error("Failed to get Alpaca snapshots - will retry in 5 seconds")
                time.sleep(5)
                return

            # Merge Massive data (prev_volume, prev_close) into Alpaca snapshots
            enriched_snapshots = self._merge_massive_into_alpaca(snapshots)

            self.candidates = self.gap_calc.find_candidates(enriched_snapshots)

            # FIX: Use config.MAX_POSITIONS instead of hardcoded 20
            self.candidates = self.gap_calc.select_by_liquidity_and_gap(
                self.candidates, max_positions=config.MAX_POSITIONS
            )

            self.vix_level = self.vix_fetcher.get_vix_level() or 15.0

            logger.info(f"Candidates found: {len(self.candidates)}")
            for c in self.candidates[:5]:
                logger.info(f"  {c.symbol}: {c.gap_pct:+.1f}% gap, ${c.adv_estimate/1e6:.1f}M ADV")

            self.stage_candidates_done = True
            self._save_state()

        except Exception as e:
            logger.error(f"Error in Step 2: {e}")
            time.sleep(5)

    def _merge_massive_into_alpaca(self, alpaca_snapshots: Dict[str, dict]) -> Dict[str, dict]:
        """Merge Massive prev_volume and prev_close into Alpaca snapshots."""
        if not self.massive_snapshots:
            logger.warning("No Massive snapshots available for merging - using Alpaca data only")
            return alpaca_snapshots

        merged = {}
        for symbol, alpaca_data in alpaca_snapshots.items():
            massive_data = self.massive_snapshots.get(symbol, {})
            
            merged[symbol] = dict(alpaca_data)
            
            # Override with Massive prev_volume and prev_close if available
            if massive_data.get("prev_volume"):
                merged[symbol]["prev_volume"] = massive_data["prev_volume"]
            if massive_data.get("prev_close"):
                merged[symbol]["prev_close"] = massive_data["prev_close"]
            
            # Fallback: Use Massive price as last_price if Alpaca doesn't have it
            if not merged[symbol].get("last_price") and massive_data.get("price"):
                merged[symbol]["last_price"] = massive_data["price"]

        logger.info(f"Merged Massive data into {len(merged)} Alpaca snapshots")
        return merged

    def _step3_enter_positions(self):
        """Step 3: Enter market orders for candidates"""
        logger.info("STEP 3: Entering positions")

        if not self.candidates:
            logger.info("No candidates to enter")
            self.stage_entry_done = True
            return

        # CRITICAL FIX: Execution guard - prevent duplicate entries on crash/restart
        # Check if we already have positions (entry already completed)
        if self.position_mgr.get_position_count() > 0:
            logger.warning(f"Execution guard: {self.position_mgr.get_position_count()} positions already exist - skipping entry")
            self.stage_entry_done = True
            return

        # Check if entry was already completed (stage flag)
        if self.stage_entry_done:
            logger.info("Execution guard: stage_entry_done flag is True - skipping entry")
            return

        try:
            positions = self.position_mgr.enter_positions(self.candidates, self.vix_level)

            logger.info(f"Entered {len(positions)} positions")
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.entry_price:.2f}")

            self.stage_entry_done = True
            self._save_state()

        except Exception as e:
            logger.error(f"Error in Step 3: {e}")
            time.sleep(5)

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
