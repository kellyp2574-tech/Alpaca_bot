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
from typing import List, Optional

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

        # Explicit stage flags for robust state tracking
        self.stage_universe_done = False
        self.stage_candidates_done = False
        self.stage_entry_done = False
        self.stage_exit_done = False

        # Retry counters for API resilience
        self.universe_retry_count = 0
        self.max_universe_retries = 3

        # Timing targets
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

            # Step 1: Universe reduction via Massive (target 9:00, retry until 9:28)
            if not self.stage_universe_done and current_time >= self.target_universe:
                if current_time < self.target_candidates:
                    self._step1_build_universe()
                else:
                    logger.warning("Skipped Step 1 - past candidate window")
                    self.stage_universe_done = True

            # Step 2: Gap calculation via Alpaca (target 9:29, retry until 9:30)
            if not self.stage_candidates_done and current_time >= self.target_candidates:
                if current_time < self.target_entry:
                    self._step2_find_candidates()
                else:
                    logger.warning("Skipped Step 2 - past entry window")
                    self.stage_candidates_done = True

            # Step 3: Market entry (target 9:30, retry until 10:00)
            if not self.stage_entry_done and current_time >= self.target_entry:
                if current_time < dt_time(10, 0):
                    self._step3_enter_positions()
                else:
                    logger.warning("Skipped Step 3 - past entry retry window")
                    self.stage_entry_done = True

            # Step 4: Manage exits throughout the day (after entry until market close)
            if self.stage_entry_done and not self.stage_exit_done:
                if current_time < self.market_close:
                    self._step4_manage_exits(current_time)
                else:
                    # Market close reached - force exits BEFORE marking complete
                    logger.info("Market close reached - forcing exits before finalizing")
                    if self.position_mgr.get_position_count() > 0:
                        self.position_mgr.force_exit_all("day_end")
                    
                    # Only mark done if positions are actually closed
                    if self.position_mgr.get_position_count() == 0:
                        self.stage_exit_done = True
                    else:
                        logger.error("Market close exit attempted, but positions remain open")

            # Check if day is complete
            if current_time >= self.market_close and self.stage_entry_done and self.stage_exit_done:
                logger.info("Market closed - day complete")
                self._finalize_day()
                break

            # Sleep to avoid busy waiting
            time.sleep(1)

    def _time_matches(self, current: dt_time, target_str: str, tolerance_seconds: int = 30) -> bool:
        """Check if current time is within tolerance of target time"""
        target = datetime.strptime(target_str, "%H:%M").time()
        current_seconds = current.hour * 3600 + current.minute * 60 + current.second
        target_seconds = target.hour * 3600 + target.minute * 60 + target.second
        return abs(current_seconds - target_seconds) <= tolerance_seconds

    def _step1_build_universe(self):
        """Step 1: Pull full market snapshot from Massive, filter by price range"""
        logger.info("STEP 1: Building universe from Massive")

        try:
            # Get full market snapshot
            snapshots = self.massive.get_full_market_snapshot()
            if not snapshots:
                self.universe_retry_count += 1
                if self.universe_retry_count >= self.max_universe_retries:
                    logger.error(f"Failed to get Massive snapshot after {self.max_universe_retries} retries, falling back to Alpaca")
                    self._fallback_to_alpaca_universe()
                else:
                    logger.warning(f"Empty Massive snapshot, retry {self.universe_retry_count}/{self.max_universe_retries}")
                    time.sleep(5)  # Back off before retry
                return

            # Filter by price range ($1-$2)
            self.universe = self.massive.filter_by_price_range(
                snapshots, config.MIN_PRICE, config.MAX_PRICE
            )

            logger.info(f"Universe built: {len(self.universe)} symbols")
            self.stage_universe_done = True

            # Persist state
            self._save_state()

        except Exception as e:
            logger.error(f"Error in Step 1: {e}")
            self.universe_retry_count += 1
            if self.universe_retry_count >= self.max_universe_retries:
                logger.error(f"Max retries exceeded for Step 1, falling back to Alpaca")
                self._fallback_to_alpaca_universe()
            # Don't mark done on error - will retry next loop iteration

    def _fallback_to_alpaca_universe(self):
        """Fallback: Build universe from Alpaca assets when Massive fails"""
        logger.info("FALLBACK: Building universe from Alpaca assets")
        
        try:
            # Get tradable assets from Alpaca
            assets = self.alpaca.get_tradable_assets()
            if not assets:
                logger.error("Alpaca fallback also failed - no universe available")
                self.stage_universe_done = True  # Give up
                return
            
            # Filter by price range using Alpaca snapshots
            # Chunk through all assets until we have enough qualified names
            chunk_size = 1000
            min_target_universe = 500  # Target at least 500 symbols
            
            self.universe = []
            snapshots_received = False
            
            for i in range(0, len(assets), chunk_size):
                chunk = assets[i:i + chunk_size]
                snapshots = self.alpaca.get_snapshots(chunk)
                
                if snapshots:
                    snapshots_received = True
                
                for symbol, data in snapshots.items():
                    # Use fallback chain: last_price -> close -> prev_close
                    price = (
                        data.get("last_price")
                        or data.get("close")
                        or data.get("prev_close")
                        or 0
                    )
                    if price and config.MIN_PRICE <= price <= config.MAX_PRICE:
                        self.universe.append(symbol)
                
                # Stop early if we have enough
                if len(self.universe) >= min_target_universe:
                    logger.info(f"Fallback: reached {len(self.universe)} qualified symbols, stopping")
                    break
            
            if not snapshots_received:
                logger.error("Alpaca fallback: failed to get any snapshots")
                self.stage_universe_done = True
                return
            
            logger.info(f"Alpaca fallback universe: {len(self.universe)} symbols")
            self.stage_universe_done = True
            self._save_state()
            
        except Exception as e:
            logger.error(f"Alpaca fallback error: {e}")
            self.stage_universe_done = True  # Give up after fallback fails

    def _step2_find_candidates(self):
        """Step 2: Fetch Alpaca snapshots, compute gaps, find candidates"""
        logger.info("STEP 2: Finding gap candidates via Alpaca")

        if not self.universe:
            logger.error("No universe available for Step 2")
            time.sleep(5)  # Sleep to prevent log spam
            return

        try:
            # Get Alpaca snapshots for universe
            snapshots = self.alpaca.get_snapshots(self.universe)
            if not snapshots:
                logger.error("Failed to get Alpaca snapshots - will retry in 5 seconds")
                time.sleep(5)  # Sleep to avoid spam
                return

            # Find gap candidates
            self.candidates = self.gap_calc.find_candidates(snapshots)

            # Select top candidates by liquidity
            self.candidates = self.gap_calc.select_by_liquidity_and_gap(
                self.candidates, max_positions=20
            )

            # Get VIX for exit regime
            self.vix_level = self.vix_fetcher.get_vix_level() or 15.0

            logger.info(f"Candidates found: {len(self.candidates)}")
            for c in self.candidates[:5]:
                logger.info(f"  {c.symbol}: {c.gap_pct:+.1f}% gap, ${c.adv_estimate/1e6:.1f}M ADV")

            self.stage_candidates_done = True

            # Persist state
            self._save_state()

        except Exception as e:
            logger.error(f"Error in Step 2: {e}")
            time.sleep(5)  # Sleep to avoid spam on error

    def _step3_enter_positions(self):
        """Step 3: Enter market orders for candidates"""
        logger.info("STEP 3: Entering positions")

        if not self.candidates:
            logger.info("No candidates to enter")
            self.stage_entry_done = True
            return

        try:
            # Enter positions
            positions = self.position_mgr.enter_positions(self.candidates, self.vix_level)

            logger.info(f"Entered {len(positions)} positions")
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.entry_price:.2f}")

            self.stage_entry_done = True
            self._save_state()

        except Exception as e:
            logger.error(f"Error in Step 3: {e}")
            time.sleep(5)  # Back off before retry

    def _step4_manage_exits(self, current_time: dt_time):
        """Step 4: Monitor and execute exits based on VIX regime"""
        if self.position_mgr.get_position_count() == 0:
            logger.info("All positions closed - exit complete")
            self.stage_exit_done = True
            return

        # Update position states (peak prices, trailing stops) - returns current prices
        current_prices = self.position_mgr.update_positions()

        # Check for exits (pass current_prices to avoid double API calls)
        exited = self.position_mgr.check_exits(current_time, self.vix_level, current_prices)
        if exited:
            logger.info(f"Exited {len(exited)} positions: {exited}")

        # Persist state
        self._save_state()

    def _load_state(self):
        """Load state from previous run (for recovery)"""
        positions = self.state_mgr.load_positions()
        if positions:
            logger.info(f"Loaded {len(positions)} positions from state, restoring...")
            self.position_mgr.load_positions(positions)
            self.stage_entry_done = True  # Assume entry was done if we have positions
            self.stage_universe_done = True
            self.stage_candidates_done = True
        
        # Load bot state (VIX, stages) - verify date matches today
        bot_state = self.state_mgr.load_bot_state()
        today = datetime.now().strftime("%Y-%m-%d")
        if bot_state:
            saved_date = bot_state.get("date")
            if saved_date == today:
                self.vix_level = bot_state.get("vix_level", self.vix_level)
                self.stage_universe_done = bot_state.get("stage_universe_done", self.stage_universe_done)
                self.stage_candidates_done = bot_state.get("stage_candidates_done", self.stage_candidates_done)
                self.stage_entry_done = bot_state.get("stage_entry_done", self.stage_entry_done)
                self.stage_exit_done = bot_state.get("stage_exit_done", self.stage_exit_done)
                logger.info(f"Restored state: VIX={self.vix_level}, stages: universe={self.stage_universe_done}, candidates={self.stage_candidates_done}, entry={self.stage_entry_done}, exit={self.stage_exit_done}")
            else:
                logger.warning(f"Stale bot state from {saved_date} - starting fresh (today is {today})")
                self.state_mgr.clear_bot_state()  # Clear stale state

    def _save_state(self):
        """Save current state"""
        self.state_mgr.save_positions(self.position_mgr.positions)
        
        # Save bot state including VIX and date
        today = datetime.now().strftime("%Y-%m-%d")
        self.state_mgr.save_bot_state({
            "date": today,
            "vix_level": self.vix_level,
            "stage_universe_done": self.stage_universe_done,
            "stage_candidates_done": self.stage_candidates_done,
            "stage_entry_done": self.stage_entry_done,
            "stage_exit_done": self.stage_exit_done,
        })

    def _finalize_day(self):
        """Finalize trading day, log summary"""
        logger.info("Finalizing trading day")

        # Force close any remaining positions
        if self.position_mgr.get_position_count() > 0:
            logger.warning("Force exiting remaining positions")
            self.position_mgr.force_exit_all("day_end")
        
        # Verify positions were actually closed before clearing state
        if self.position_mgr.get_position_count() > 0:
            logger.critical("FAILED TO FLATTEN POSITIONS AT END OF DAY - preserving state")
            return  # Do NOT clear state if positions remain

        # Clear state only after successful exit
        self.state_mgr.clear_positions()
        self.state_mgr.clear_bot_state()

        # Log summary
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
    """Entry point"""
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
