"""
Gap Momentum Trading Bot
Runs from 8:30 AM until all positions are closed (hard exit at 2:30 PM)
"""
import sys
import logging
import os
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Import bot components
from bot import config as ma_config
from bot import alpaca_client as broker
from bot.trade_reporter import get_trade_reporter
from bot.reporting_position_manager import create_reporting_position_manager

# Import morning momentum components (now local)
from bot.morning_config import Config as MMConfig
from bot.data_sources import init_data_stack, DataStack
from bot.execution import ExecutionClient, ExecutionConfig
from bot.position_manager import PositionManager
from bot.risk_manager import RiskManager
from bot.state_manager import StateStore as MMStateStore
from bot.morning_main import EntryContext, EntryLoop, _reconcile_pending_entries
from bot.clock import market_datetime, market_now, config_window, MARKET_TZ

# ═══════════════════════════════════════════════════
# Global Logging setup - all modules will use this
# ═══════════════════════════════════════════════════
os.makedirs(ma_config.LOG_DIR, exist_ok=True)

LOG_PATH = ma_config.LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("integrated_bot")
logger.info(f"Logging initialized: {LOG_PATH}")


class IntegratedBot:
    """Gap momentum trading bot - monitors positions until all closed"""
    
    def __init__(self, dry_run=False):
        logger.info("IntegratedBot.__init__ starting...")
        self.dry_run = dry_run
        self.mm_config = MMConfig()
        self.mm_state_store = MMStateStore("state/mm_positions.json")
        
        # Initialize trade reporter with error handling (don't let reporting kill trading)
        try:
            self.trade_reporter = get_trade_reporter()
            logger.info("TradeReporter initialized successfully")
        except Exception as e:
            logger.exception("TradeReporter init failed; disabling reporting. Error=%s", e)
            self.trade_reporter = None
        
        # Initialize data stack
        self.mm_data = init_data_stack()
        
        # Initialize execution client
        exec_cfg = ExecutionConfig(
            buy_slippage_pct=self.mm_config.exec_slippage_buy_pct,
            sell_slippage_pct=self.mm_config.exec_slippage_sell_pct,
        )
        self.execution = ExecutionClient(
            paper=ma_config.ALPACA_PAPER,
            dry_run=dry_run,
            cfg=exec_cfg,
            quote_provider=self.mm_data.alpaca.get_latest_quote,
        )
        
        # Risk manager for morning momentum
        self.risk_manager = RiskManager(self.mm_config, state_store=self.mm_state_store)
        
        self.momentum_completed = False
        self.mm_positions = None
        self.mm_execution = None
        self.clock_check_failed = False
        
        # Candidate caching to prevent rescans on retry
        self.mm_candidates = None
        self.mm_watchlist = None
        self.mm_subscribe_symbols = None
        self.mm_candidate_map = None
        self.mm_candidates_date = None
        
        # Stage-specific caching to avoid rebuilding stage 1 on every call
        self.mm_stage1_result = None
        self.mm_stage2_result = None
        self.mm_stage3_result = None
        self.mm_stages_completed = set()  # Track which stages have been run
    
    def _check_orphaned_broker_positions(self):
        """Check for and log any unexpected positions in broker account."""
        try:
            positions = broker.get_all_positions()
            if not positions:
                return
            
            # Known tickers from both strategies
            ma_tickers = {
                ma_config.MA_TRADE_GROWTH,
                ma_config.MA_TRADE_SAFE,
                ma_config.MA_TRADE_ALT,
            }
            
            # Get MM positions if available
            mm_symbols = set()
            if self.mm_positions:
                mm_symbols = set(self.mm_positions.positions.keys())
            
            known_symbols = ma_tickers | mm_symbols
            
            # Check for orphaned positions
            for pos in positions:
                symbol = getattr(pos, "symbol", None)
                qty = float(getattr(pos, "qty", 0) or 0)
                if symbol and abs(qty) > 0 and symbol not in known_symbols:
                    logger.warning(f"⚠️ ORPHANED POSITION DETECTED: {symbol} qty={qty} - not tracked by bot")
        except Exception as e:
            logger.error(f"Failed to check orphaned positions: {e}")
        
    def run(self):
        """Main bot loop from 8:30 AM until all positions closed"""
        logger.info("=" * 60)
        logger.info("INTEGRATED BOT START" + (" [DRY RUN]" if self.dry_run else ""))
        logger.info("=" * 60)
        
        try:
            # Check if market is open (but allow morning momentum to start at 8:30)
            clock = broker.get_clock()
            if not clock.is_open and not self.dry_run:
                now = market_now()
                universe_build_time = now.replace(hour=8, minute=30, second=0, microsecond=0)
                
                # If it's after 8:30 AM but before market opens, allow morning momentum to start
                if now.time() >= datetime.strptime("08:30", "%H:%M").time():
                    logger.info("Market not open yet, but starting morning momentum at 8:30 AM")
                    # Continue to main loop to start staged candidate building
                elif now.time() >= datetime.strptime("08:00", "%H:%M").time():
                    # Between 8:00-8:30: wait for 8:30
                    wait_seconds = (universe_build_time - now).total_seconds()
                    if wait_seconds > 0:
                        logger.info(f"Waiting for morning momentum start at 8:30 AM ({wait_seconds/60:.1f} minutes)")
                        time.sleep(wait_seconds)
                else:
                    logger.info("Market is closed and it's not within operating hours. Exiting.")
                    return
        except Exception as e:
            logger.warning(f"Could not check market clock: {e}. Continuing in safe mode (no trading until confirmed open)")
            # Don't exit - continue running but be cautious
            self.clock_check_failed = True
        
        # Main loop
        while True:
            now = market_now()
            current_time = now.time()
            
            # Safety exit at 3:30 PM (should have exited earlier when positions closed)
            if current_time >= datetime.strptime("15:30", "%H:%M").time():
                logger.warning("Reached 3:30 PM safety exit - positions may still be open")
                break
            
            # If market has closed early, exit gracefully (avoid false negatives at the open)
            if current_time >= datetime.strptime("09:35", "%H:%M").time() and not self.dry_run:
                try:
                    clock = broker.get_clock()
                    if not clock.is_open:
                        # require confirmation before exiting
                        if not hasattr(self, "_closed_clock_first_seen"):
                            self._closed_clock_first_seen = time.monotonic()
                            logger.warning(
                                "Clock reports closed after 9:35; will re-check before exiting. "
                                f"next_open={getattr(clock,'next_open',None)} next_close={getattr(clock,'next_close',None)}"
                            )
                        elif time.monotonic() - self._closed_clock_first_seen > 60:
                            logger.info("Market clock still closed after confirm window; shutting down loop")
                            break
                    else:
                        # reset if clock is healthy again
                        if hasattr(self, "_closed_clock_first_seen"):
                            delattr(self, "_closed_clock_first_seen")
                except Exception as e:
                    logger.warning(f"Failed to check market clock during run loop: {e}")

            # If clock check failed, be extra cautious before 9:30 AM
            if self.clock_check_failed and current_time < datetime.strptime("09:30", "%H:%M").time():
                logger.info("Clock check failed - waiting until 9:30 AM to ensure market is open")
                time.sleep(60)
                continue
            
            # Re-check clock status after 9:30 AM if it failed before
            if self.clock_check_failed and current_time >= datetime.strptime("09:30", "%H:%M").time():
                try:
                    clock = broker.get_clock()
                    if clock.is_open:
                        logger.info("Clock re-check successful - market is open, proceeding normally")
                        self.clock_check_failed = False
                    else:
                        logger.warning("Clock re-check shows market still closed - continuing to wait")
                        time.sleep(60)
                        continue
                except Exception as e:
                    logger.warning(f"Clock re-check failed: {e} - continuing to wait")
                    time.sleep(60)
                    continue
            
            # Morning Momentum: Staged timeline 8:30 AM - hard_exit + 30 min cleanup buffer
            mm_cleanup_deadline = datetime.strptime(self.mm_config.hard_exit, "%H:%M").time()
            mm_cleanup_deadline = (datetime.combine(datetime.today(), mm_cleanup_deadline) + timedelta(minutes=30)).time()
            if current_time < mm_cleanup_deadline:
                if not self.momentum_completed:
                    # If after entry cutoff, skip morning momentum (entry window closed)
                    if current_time >= datetime.strptime(self.mm_config.entry_cutoff, "%H:%M").time():
                        logger.info(f"Started after {self.mm_config.entry_cutoff} - morning momentum entry window closed, skipping")
                        self.momentum_completed = True
                        continue
                    
                    # Staged timeline: Run at appropriate times (8:30, 9:05, 9:15, 9:25)
                    universe_build = datetime.strptime(self.mm_config.universe_build_time, "%H:%M").time()
                    
                    # If before 8:30, wait for universe build time
                    if current_time < universe_build:
                        wait_time = now.replace(hour=universe_build.hour, minute=universe_build.minute, second=0, microsecond=0)
                        wait_seconds = (wait_time - now).total_seconds()
                        if wait_seconds > 0:
                            logger.info(f"Waiting for universe build at {self.mm_config.universe_build_time} ({wait_seconds/60:.1f} minutes)")
                            time.sleep(min(wait_seconds, 60))  # Check every minute
                            continue
                    
                    # Run morning momentum strategy (handles staged timeline internally)
                    status = self.run_morning_momentum(now)
                    if status == "completed":
                        self.momentum_completed = True
                        logger.info("Morning momentum completed successfully")
                    elif status == "in_progress":
                        # Stage completed, continue normal loop timing (no error, no backoff)
                        logger.debug("Morning momentum stage completed, continuing...")
                        time.sleep(5)  # Brief sleep to avoid tight loop
                    elif status == "failed":
                        logger.error("Morning momentum failed; waiting 30 seconds before retry")
                        time.sleep(30)  # Backoff to prevent rapid retry loops
                    else:
                        logger.error(f"Unexpected status from run_morning_momentum: {status}")
                        time.sleep(30)
                else:
                    # After momentum completes, supervise positions until hard exit
                    self._supervise_mm_positions_until_hard_exit(now)
                    time.sleep(30)  # Prevent tight loop after supervision returns
                    continue
            
            # Check for any orphaned broker positions after emergency flatten
            self._check_orphaned_broker_positions()
            
            # After hard exit: Monitor positions every 5 minutes until all closed
            hard_exit_time = datetime.strptime(self.mm_config.hard_exit, "%H:%M").time()
            if current_time >= hard_exit_time:
                positions_closed = self._all_positions_closed()
                
                if positions_closed:
                    logger.info("All positions closed - waiting for 4:05 PM to generate liquidity ranking")
                    
                    # Sleep until 4:05 PM for liquidity ranking generation
                    ranking_time = now.replace(hour=16, minute=5, second=0, microsecond=0)
                    if now < ranking_time:
                        wait_seconds = (ranking_time - now).total_seconds()
                        logger.info(f"Sleeping for {wait_seconds/60:.1f} minutes until 4:05 PM")
                        time.sleep(wait_seconds)
                    
                    # Generate liquidity ranking for tomorrow's universe
                    # This runs regardless of position state to ensure daily refresh
                    logger.info("Generating liquidity ranking at 4:05 PM...")
                    self._generate_liquidity_ranking()
                    
                    logger.info("Liquidity ranking complete - shutting down")
                    break
                else:
                    # If positions still open past 3:00 PM, generate ranking anyway at 4:05 PM
                    # to ensure daily refresh even if there's a position state issue
                    if current_time >= datetime.strptime("15:00", "%H:%M").time():
                        logger.warning("Positions still open past 3:00 PM - will generate ranking at 4:05 PM anyway")
                        
                        ranking_time = now.replace(hour=16, minute=5, second=0, microsecond=0)
                        if now < ranking_time:
                            wait_seconds = (ranking_time - now).total_seconds()
                            logger.info(f"Sleeping for {wait_seconds/60:.1f} minutes until 4:05 PM")
                            time.sleep(wait_seconds)
                        
                        logger.info("Generating liquidity ranking at 4:05 PM (positions may still be open)...")
                        self._generate_liquidity_ranking()
                        
                        logger.warning("Liquidity ranking complete - shutting down with positions potentially open")
                        break
                    else:
                        logger.info("Positions still open - checking again in 5 minutes")
                        time.sleep(300)  # 5 minutes
    
    def _supervise_mm_positions_until_hard_exit(self, now):
        """Supervise MM positions until hard exit time, regardless of EntryLoop status."""
        hard_exit_time = datetime.strptime(self.mm_config.hard_exit, "%H:%M").time()
        
        if now.time() >= hard_exit_time:
            logger.info("Hard exit time reached - forcing MM positions flat")
            self._force_mm_positions_flat()
            return
        
        # Check if we have any MM positions to supervise
        try:
            mm_positions = self.mm_state_store.load_positions()
            if not mm_positions and self.mm_positions is not None:
                mm_positions = self.mm_positions.positions

            if not mm_positions:
                logger.info("No MM positions to supervise - waiting for hard exit")
                # Sleep until hard exit time
                hard_exit_dt = now.replace(
                    hour=int(self.mm_config.hard_exit.split(':')[0]),
                    minute=int(self.mm_config.hard_exit.split(':')[1]),
                    second=0, microsecond=0
                )
                wait_seconds = (hard_exit_dt - now).total_seconds()
                if wait_seconds > 0:
                    logger.info(f"Waiting for hard exit at {hard_exit_time} ({wait_seconds/60:.1f} minutes)")
                    time.sleep(min(wait_seconds, 300))  # Cap at 5 minutes, will re-evaluate
                return
            
            logger.info(f"Supervising {len(mm_positions)} MM positions until hard exit at {hard_exit_time}")
            
            # Check exits periodically (every 30 seconds)
            while now.time() < hard_exit_time:
                try:
                    # Use self.mm_positions.positions as single source of truth
                    if self.mm_positions is not None:
                        current_positions = self.mm_positions.positions
                    else:
                        current_positions = self.mm_state_store.load_positions()
                        if not current_positions:
                            logger.info("All MM positions closed - waiting for hard exit")
                            break
                    if not current_positions:
                        logger.info("All MM positions closed - waiting for hard exit")
                        break
                    
                    # Get current quotes for exit price checks (batched)
                    symbols = list(current_positions.keys())
                    quotes = {}
                    try:
                        quote_dict = self.mm_data.alpaca.get_latest_quotes(symbols)
                        if quote_dict:
                            quotes = quote_dict
                    except Exception as e:
                        logger.warning(f"Failed to get batched quotes for MM supervision: {e}")
                    
                    # Check position exits using quotes (not calling on_bar with Quote objects)
                    for symbol in list(current_positions.keys()):
                        if symbol in quotes:
                            quote = quotes[symbol]
                            position = current_positions.get(symbol)
                            if position and quote.bid_price > 0:
                                # Check if stop loss hit
                                if quote.bid_price <= position.stop_price:
                                    logger.info(f"Stop loss hit for {symbol}: bid={quote.bid_price:.2f} <= stop={position.stop_price:.2f}")
                                    if self.mm_positions is not None:
                                        self.mm_positions.exit_position(symbol, quote.bid_price, market_now(), reason="stop_loss")
                                    else:
                                        logger.warning("MM supervisor: no active position manager, forcing broker close for %s", symbol)
                                        order = self.execution.client.close_position(symbol)
                                        try:
                                            stored_positions = self.mm_state_store.load_positions()
                                        except RuntimeError as err:
                                            logger.critical(
                                                "Unable to load MM state while supervising %s broker close: %s",
                                                symbol,
                                                err,
                                            )
                                            stored_positions = {}

                                        state = stored_positions.get(symbol)
                                        if state is not None:
                                            state.exit_pending = True
                                            state.exit_reason = "supervisor_broker_close"
                                            state.exit_submitted_ts = time.monotonic()
                                            state.exit_order_id = getattr(order, "id", state.exit_order_id)
                                            if getattr(order, "client_order_id", None):
                                                state.exit_client_order_id = order.client_order_id
                                            self.mm_state_store.save_positions(stored_positions)
                                        else:
                                            logger.warning(
                                                "Supervisor broker close for %s without stored position; state may already be cleared",
                                                symbol,
                                            )
                    
                    # Sleep for 30 seconds
                    time.sleep(30)
                    now = market_now()
                    
                except Exception as e:
                    logger.error(f"Error in MM position supervision: {e}")
                    time.sleep(30)
                    now = market_now()
            
            # Force flat at hard exit time regardless
            if self.mm_positions is not None:
                logger.info("Hard exit time reached - forcing MM positions flat")
                self._force_mm_positions_flat()
            else:
                logger.warning("MM supervisor: no active position manager, skipping force flat")
            
        except Exception as e:
            logger.error(f"Critical error in MM position supervision: {e}")
            # Emergency flatten on error
            if self.mm_positions is not None:
                self._force_mm_positions_flat()
            else:
                logger.warning("MM supervisor: no active position manager, skipping emergency flatten")
    
    def _force_mm_positions_flat(self):
        """Force all MM positions flat immediately."""
        try:
            logger.warning("EMERGENCY: Forcing all MM positions flat")
            
            # Cancel all MM orders first
            self._cancel_all_mm_orders()
            
            runtime_manager = self.mm_positions
            runtime_positions = {}
            if runtime_manager is not None:
                runtime_positions = dict(runtime_manager.positions)

            try:
                stored_positions = self.mm_state_store.load_positions()
            except RuntimeError as e:
                logger.critical(f"Unable to load stored MM positions during emergency flatten: {e}")
                stored_positions = {}

            allowed_symbols = set(runtime_positions.keys()) | set(stored_positions.keys())

            if not allowed_symbols:
                logger.info("No MM positions recorded; nothing to flatten")
                return

            # Get current prices for all tracked symbols (batched)
            price_lookup = {}
            symbols = list(allowed_symbols)
            
            try:
                quote_dict = self.mm_data.alpaca.get_latest_quotes(symbols)
            except Exception as e:
                logger.warning(f"Failed to get batched quotes during emergency flatten: {e}")
                quote_dict = {}
            
            for symbol in symbols:
                quote = quote_dict.get(symbol) if quote_dict else None
                if quote and getattr(quote, "bid_price", 0) > 0:
                    price_lookup[symbol] = quote.bid_price
                else:
                    position = runtime_positions.get(symbol) or stored_positions.get(symbol)
                    if position:
                        price_lookup[symbol] = position.peak_price or position.entry_price

            # Force exits using runtime manager when available, otherwise broker fallback
            closed_symbols: set[str] = set()

            if runtime_manager is not None and runtime_manager.positions:
                before_symbols = set(runtime_manager.positions.keys())
                runtime_manager.force_exit_all(price_lookup, reason="emergency_hard_exit")

                logger.info("Starting MM time-based exit reconciliation...")
                runtime_manager.reconcile_pending_exits_time_based(max_wait_seconds=30.0)

                remaining_symbols = set(runtime_manager.positions.keys())
                closed_symbols.update(before_symbols - remaining_symbols)

                if remaining_symbols:
                    logger.error(
                        "CRITICAL: %d MM positions still open after emergency flatten via manager",
                        len(remaining_symbols),
                    )
                    self._emergency_mm_flatten()
            else:
                client = getattr(self.mm_execution, "client", None) or getattr(self.execution, "client", None)
                if client is None:
                    logger.critical("No execution client available for emergency flatten fallback")
                for symbol in allowed_symbols:
                    try:
                        logger.critical(f"Emergency closing MM position (fallback) {symbol}")
                        order = None
                        if client:
                            order = client.close_position(symbol)

                        state = (
                            runtime_positions.get(symbol)
                            or stored_positions.get(symbol)
                        )
                        if state is None:
                            logger.warning(
                                "Fallback emergency close for %s without stored state; tracking skipped",
                                symbol,
                            )
                            continue

                        state.exit_reason = "integrated_emergency_close"
                        state.exit_pending = True
                        state.exit_submitted_ts = time.monotonic()
                        state.exit_time = None
                        if order is not None:
                            state.exit_order_id = getattr(order, "id", state.exit_order_id)
                            client_id = getattr(order, "client_order_id", None)
                            if client_id:
                                state.exit_client_order_id = client_id

                        stored_positions[symbol] = state
                        if runtime_manager is not None and symbol in runtime_manager.positions:
                            runtime_manager.positions[symbol] = state
                    except Exception as e:
                        logger.error(f"Failed to emergency close {symbol}: {e}")

            # Persist cleared state
            if runtime_manager is not None:
                stored_positions = dict(runtime_manager.positions)
            self.mm_state_store.save_positions(stored_positions)
            
        except Exception as e:
            logger.error(f"Critical error forcing MM positions flat: {e}")
    
    def _cancel_all_mm_orders(self):
        """Cancel all MM-related orders."""
        try:
            client = getattr(self.mm_execution, "client", None) or getattr(self.execution, "client", None)
            if client:
                orders = client.get_orders()
                mm_symbols = set()
                if self.mm_positions is not None:
                    mm_symbols.update(self.mm_positions.positions.keys())
                try:
                    mm_symbols.update(self.mm_state_store.load_positions().keys())
                except Exception:
                    pass
                
                for order in orders:
                    order_status = getattr(order, 'status', None)
                    client_order_id = getattr(order, 'client_order_id', None)
                    symbol = getattr(order, 'symbol', None)
                    
                    # Cancel any order that's still open
                    if (
                        order_status in {'new', 'partially_filled', 'submitted', 'accepted'}
                        and symbol in mm_symbols
                        and client_order_id
                        and (
                            client_order_id.startswith("ENTRY:")
                            or client_order_id.startswith("EXIT:")
                            or client_order_id.startswith("MM:")
                        )
                    ):
                        try:
                            client.cancel_order(order.id)
                            logger.info(f"Cancelled MM order {order.id} for {symbol} ({client_order_id})")
                            
                            # Clear pending entry state if it's an entry order
                            if client_order_id and client_order_id.startswith("ENTRY:"):
                                self.mm_state_store.clear_pending_entry(client_order_id)
                                
                        except Exception as e:
                            logger.warning(f"Failed to cancel MM order {order.id}: {e}")
                            
        except Exception as e:
            logger.warning(f"Error cancelling MM orders: {e}")
    
    def _emergency_mm_flatten(self):
        """Emergency flatten for MM positions - close everything at market."""
        try:
            logger.critical("EMERGENCY MM FLATTEN: Closing all positions at market")
            
            client = getattr(self.mm_execution, "client", None) or getattr(self.execution, "client", None)
            if client is None:
                logger.critical("No execution client available for emergency MM flatten")
                return

            # Get all positions from broker
            positions = client.get_positions()
            allowed_symbols = set()
            if self.mm_positions is not None:
                allowed_symbols.update(self.mm_positions.positions.keys())
            try:
                stored_positions = self.mm_state_store.load_positions()
                allowed_symbols.update(stored_positions.keys())
            except Exception:
                stored_positions = {}
            
            for pos in positions:
                symbol = getattr(pos, 'symbol', None)
                qty = float(getattr(pos, 'qty', 0) or 0)

                if symbol and qty > 0 and symbol in allowed_symbols:
                    try:
                        logger.critical(f"Emergency closing MM position {symbol} ({qty} shares)")
                        order = client.close_position(symbol)

                        state = None
                        if self.mm_positions is not None:
                            state = self.mm_positions.positions.get(symbol)
                        if state is None:
                            state = stored_positions.get(symbol)
                        if state is None:
                            logger.warning("Emergency flatten: no state record for %s; unable to track close", symbol)
                            continue

                        state.exit_reason = "integrated_emergency_close"
                        state.exit_pending = True
                        state.exit_submitted_ts = time.monotonic()
                        state.exit_time = None
                        state.exit_order_id = getattr(order, "id", state.exit_order_id)
                        client_id = getattr(order, "client_order_id", None)
                        if client_id:
                            state.exit_client_order_id = client_id

                        stored_positions[symbol] = state
                        if self.mm_positions is not None:
                            self.mm_positions.positions[symbol] = state
                        
                    except Exception as e:
                        logger.error(f"Failed to emergency close {symbol}: {e}")

            # Save state
            self.mm_state_store.save_positions(stored_positions)
            
        except Exception as e:
            logger.error(f"Critical error in emergency MM flatten: {e}")

    def run_morning_momentum(self, now) -> str:
        """Run morning momentum strategy with staged timeline.
        
        Returns:
            "in_progress": Stage completed, more stages pending
            "completed": All stages done, entry loop started
            "failed": Error occurred
        """
        logger.info("Starting morning momentum strategy (staged timeline)")

        try:
            from .premarket_scan_staged import (
                stage1_broad_filter_delayed_sip,
                stage2_first_iex_refinement,
                stage3_second_iex_refinement,
            )
            from .morning_main_staged import wait_for_timeline_stage
            
            # Check if we need to reset for a new day
            today = now.date()
            if self.mm_candidates_date != today:
                logger.info("New day detected - resetting stage cache")
                self.mm_stage1_result = None
                self.mm_stage2_result = None
                self.mm_stage3_result = None
                self.mm_stages_completed = set()
                self.mm_candidates_date = today
            
            # Determine current stage based on time
            current_time = now.time()
            stage1_time = datetime.strptime(self.mm_config.broad_filter_start, "%H:%M").time()
            stage2_time = datetime.strptime(self.mm_config.first_refinement, "%H:%M").time()
            stage3_time = datetime.strptime(self.mm_config.second_refinement, "%H:%M").time()
            freeze_time = datetime.strptime(self.mm_config.candidate_freeze, "%H:%M").time()
            
            # Stage 1: 8:30-8:40 broad filter (run once)
            if current_time >= stage1_time and 1 not in self.mm_stages_completed:
                logger.info("Running Stage 1: Broad filter (delayed_sip) at %s", current_time.strftime('%H:%M'))
                
                # Build universe from Alpaca Assets API (cached)
                from .universe_loader import build_universe
                
                logger.info("Building 4,000-symbol universe from Alpaca Assets API...")
                seed_symbols = build_universe(
                    broker,
                    target_size=self.mm_config.max_seed_universe,
                )
                
                if not seed_symbols:
                    logger.error("Failed to build universe, cannot proceed with Stage 1")
                    return "failed"
                
                logger.info(f"Universe built: {len(seed_symbols)} symbols from Alpaca Assets")
                
                # Run stage 1 with pre-built universe
                result1 = stage1_broad_filter_delayed_sip(
                    self.mm_config,
                    self.mm_data.alpaca,
                    seed_symbols,
                    now,
                )
                self.mm_stage1_result = result1.candidates
                self.mm_stages_completed.add(1)
                logger.info(f"Stage 1 complete: {len(self.mm_stage1_result)} candidates cached")
                # Return to loop - don't block waiting for next stage
                return "in_progress"  # Stage 1 done, more stages pending
            
            # Stage 2: 9:05 first IEX refinement (run once)
            if current_time >= stage2_time and 2 not in self.mm_stages_completed:
                if 1 not in self.mm_stages_completed:
                    logger.warning("Stage 2 triggered but Stage 1 not complete - running Stage 1 first")
                    
                    # Build universe from Alpaca Assets API
                    from .universe_loader import build_universe
                    
                    seed_symbols = build_universe(
                        broker,
                        target_size=self.mm_config.max_seed_universe,
                    )
                    
                    if not seed_symbols:
                        logger.error("Failed to build universe for recovery Stage 1")
                        return "failed"
                    
                    # Run stage 1 first if missed
                    result1 = stage1_broad_filter_delayed_sip(
                        self.mm_config,
                        self.mm_data.alpaca,
                        seed_symbols,
                        now,
                    )
                    self.mm_stage1_result = result1.candidates
                    self.mm_stages_completed.add(1)
                
                logger.info("Running Stage 2: First IEX refinement at %s", current_time.strftime('%H:%M'))
                result2 = stage2_first_iex_refinement(
                    self.mm_config,
                    self.mm_data.alpaca,
                    self.mm_stage1_result,
                )
                self.mm_stage2_result = result2.candidates
                self.mm_stages_completed.add(2)
                logger.info(f"Stage 2 complete: {len(self.mm_stage2_result)} candidates cached")
                # Return to loop - don't block waiting for next stage
                return "in_progress"  # Stage 2 done, more stages pending
            
            # Stage 3: 9:15 second IEX refinement (run once)
            if current_time >= stage3_time and 3 not in self.mm_stages_completed:
                if 2 not in self.mm_stages_completed:
                    logger.warning("Stage 3 triggered but Stage 2 not complete - running stages in order")
                    # Run missing stages first
                    if 1 not in self.mm_stages_completed:
                        # Build universe from Alpaca Assets API
                        from .universe_loader import build_universe
                        
                        seed_symbols = build_universe(
                            broker,
                            target_size=self.mm_config.max_seed_universe,
                        )
                        
                        if not seed_symbols:
                            logger.error("Failed to build universe for recovery Stage 1")
                            return "failed"
                        
                        result1 = stage1_broad_filter_delayed_sip(
                            self.mm_config,
                            self.mm_data.alpaca,
                            seed_symbols,
                            now,
                        )
                        self.mm_stage1_result = result1.candidates
                        self.mm_stages_completed.add(1)
                    
                    result2 = stage2_first_iex_refinement(
                        self.mm_config,
                        self.mm_data.alpaca,
                        self.mm_stage1_result,
                    )
                    self.mm_stage2_result = result2.candidates
                    self.mm_stages_completed.add(2)
                
                logger.info("Running Stage 3: Second IEX refinement at %s", current_time.strftime('%H:%M'))
                result3 = stage3_second_iex_refinement(
                    self.mm_config,
                    self.mm_data.alpaca,
                    self.mm_stage2_result,
                )
                self.mm_stage3_result = result3.candidates
                self.mm_stages_completed.add(3)
                logger.info(f"Stage 3 complete: {len(self.mm_stage3_result)} candidates cached")
                # Return to loop - don't block waiting for freeze
                return "in_progress"  # Stage 3 done, waiting for freeze
            
            # Candidate freeze: 9:25 - only proceed if we've reached freeze time
            if current_time < freeze_time:
                logger.info("Waiting for candidate freeze at %s (returning to loop)", self.mm_config.candidate_freeze)
                return "in_progress"  # Waiting for freeze time
            
            # Use final stage results
            if 3 in self.mm_stages_completed:
                candidates = self.mm_stage3_result
            elif 2 in self.mm_stages_completed:
                candidates = self.mm_stage2_result
            elif 1 in self.mm_stages_completed:
                candidates = self.mm_stage1_result
            else:
                logger.error("No stages completed - cannot proceed")
                return "failed"
            
            if not candidates:
                logger.warning("No morning momentum candidates found")
                return "completed"  # No candidates but not an error
            
            logger.info("Candidates frozen at %s", self.mm_config.candidate_freeze)
            
            # Cache final results for entry loop
            self.mm_candidates = candidates
            self.mm_watchlist = candidates[:self.mm_config.max_candidates_monitored]
            self.mm_subscribe_symbols = [c.symbol for c in candidates[:self.mm_config.max_subscribe_symbols]]
            self.mm_candidate_map = {c.symbol: c for c in candidates[:self.mm_config.max_subscribe_symbols]}
            
            logger.info(f"Final candidates: {len(candidates)} total")
            logger.info(f"Watchlist: {len(self.mm_watchlist)} symbols")
            logger.info(f"Subscribe: {len(self.mm_subscribe_symbols)} symbols")
            logger.info(f"Stages completed: {sorted(self.mm_stages_completed)}")
            
            # Use final cached values
            watchlist = self.mm_watchlist
            subscribe_symbols = self.mm_subscribe_symbols
            candidate_map = self.mm_candidate_map
            candidates = self.mm_candidates
            
            logger.info(f"Morning momentum watchlist: {', '.join(c.symbol for c in watchlist)}")
            
            # Initialize position manager
            positions = PositionManager(
                self.mm_config, 
                self.execution, 
                self.risk_manager, 
                state_store=self.mm_state_store
            )
            
            # Wrap with reporting functionality
            positions = create_reporting_position_manager(positions, "morning_momentum")

            # Store for supervision
            self.mm_positions = positions
            self.mm_execution = self.execution
            
            # Load existing positions
            existing_positions = self.mm_state_store.load_positions()
            if existing_positions:
                positions.load_states(existing_positions)
            
            # Load risk state
            risk_payload = self.mm_state_store.load_risk_state()
            self.risk_manager.load_state(risk_payload)
            self.risk_manager.maybe_reset(now.date())
            
            # Reconcile pending entries
            _reconcile_pending_entries(self.execution, self.mm_state_store, positions)
            
            # Get account info
            try:
                equity = broker.get_equity()
                cash = broker.get_cash()
            except Exception as e:
                logger.error(f"Could not fetch account info: {e}")
                equity = 100000  # Default
                cash = equity
            
            # Create entry context
            ctx = EntryContext(
                cfg=self.mm_config,
                data=self.mm_data,
                watchlist=watchlist,
                max_bar_history=120,
                candidate_map=candidate_map,
                risk_manager=self.risk_manager,
                account_equity=equity,
                account_cash=cash,
                execution=self.execution,
                positions=positions,
                state_store=self.mm_state_store,
                subscribe_symbols=subscribe_symbols,
            )
            
            # Run entry loop
            loop = EntryLoop(ctx)
            loop.run()

            # After loop completes, mark done
            logger.info("Morning momentum loop completed")
            status = "completed"

            # Save latest state
            if self.mm_positions is not None:
                self.mm_state_store.save_positions(self.mm_positions.positions)
            self.risk_manager.persist_state()

            # Check orphaned positions post MM run
            self._check_orphaned_broker_positions()
            
            # Only unsubscribe on success to avoid resubscription churn on retry
            try:
                self.mm_data.unsubscribe_all()
                logger.info("Unsubscribed from all data feeds after successful completion")
            except Exception as e:
                logger.warning(f"Failed to unsubscribe data feeds: {e}")

        except Exception as e:
            logger.error(f"Error running morning momentum: {e}", exc_info=True)
            # Ensure supervision components reflect failure
            if self.mm_positions is None:
                logger.error("Morning momentum initialization failed before position manager setup")
            else:
                logger.error("Morning momentum encountered an error after initialization; supervision will attempt emergency flatten if needed")
            # On failure, avoid using partial state and keep subscriptions for retry
            self.mm_positions = None
            self.mm_execution = None
            logger.info("Keeping data feed subscriptions active for retry")
            return "failed"

        return status

    def _generate_liquidity_ranking(self) -> None:
        """Generate liquidity ranking file for tomorrow's universe selection."""
        try:
            from pathlib import Path
            from .liquidity_ranker import generate_liquidity_ranking
            
            output_path = Path(__file__).resolve().parents[1] / "state" / "universe" / "liquidity_ranking.json"
            
            success = generate_liquidity_ranking(
                broker,
                self.mm_data.alpaca,
                output_path,
            )
            
            if success:
                logger.info("Successfully generated liquidity ranking for tomorrow")
            else:
                logger.error("Failed to generate liquidity ranking")
        
        except Exception as e:
            logger.error(f"Error generating liquidity ranking: {e}", exc_info=True)
    
    def _all_positions_closed(self) -> bool:
        """Check if all MM positions are closed and logs are posted."""
        try:
            # Check MM positions from state store
            mm_positions = self.mm_state_store.load_positions()
            if mm_positions:
                logger.info(f"MM positions still open: {list(mm_positions.keys())}")
                return False
            
            # Check runtime position manager
            if self.mm_positions is not None and self.mm_positions.positions:
                logger.info(f"Runtime MM positions still open: {list(self.mm_positions.positions.keys())}")
                return False
            
            # Check broker positions to be safe
            try:
                positions = broker.get_all_positions()
                if positions:
                    open_symbols = [p.symbol for p in positions if float(getattr(p, 'qty', 0) or 0) > 0]
                    if open_symbols:
                        logger.info(f"Broker positions still open: {open_symbols}")
                        return False
            except Exception as e:
                logger.warning(f"Could not check broker positions: {e}")
            
            logger.info("All positions confirmed closed")
            return True
            
        except Exception as e:
            logger.error(f"Error checking if positions closed: {e}")
            return False
    
def main():
    parser = argparse.ArgumentParser(description="Integrated Bot - Morning Momentum + 3 ETF Rotation")
    parser.add_argument("--dry-run", action="store_true", help="Show signals without trading")
    args = parser.parse_args()
    
    bot = IntegratedBot(dry_run=args.dry_run)
    bot.run()


if __name__ == "__main__":
    main()
