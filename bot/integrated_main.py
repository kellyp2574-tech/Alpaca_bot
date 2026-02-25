"""
Integrated Bot - Combines 3 ETF Rotation with Morning Momentum Strategy
Runs from 8:30 AM to 3:30 PM with strategy switching at 11:00 AM
"""
import sys
import logging
import os
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Import current bot components (3 ETF rotation)
from bot import config as ma_config
from bot.state_manager import load_state, save_state, log_trade
from bot import alpaca_client as broker
from bot import data as ma_data
from bot import strategies as ma_strategies
from bot.trade_reporter import get_trade_reporter, log_trade_with_reporting
from bot.reporting_position_manager import create_reporting_position_manager

# Import morning momentum components (now local)
from bot.morning_config import Config as MMConfig
from bot.data_sources import init_data_stack, DataStack
from bot.execution import ExecutionClient, ExecutionConfig
from bot.position_manager import PositionManager
from bot.risk_manager import RiskManager
from bot.state_manager import StateStore as MMStateStore
from bot.morning_main import fetch_candidates, EntryContext, EntryLoop, _reconcile_pending_entries
from bot.clock import market_datetime, market_now, config_window, MARKET_TZ

# ═══════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════
os.makedirs(ma_config.LOG_DIR, exist_ok=True)

logger = logging.getLogger("integrated_bot")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(ma_config.LOG_FILE)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class IntegratedBot:
    """Integrated bot that runs morning momentum then 3 ETF rotation"""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.mm_config = MMConfig()
        self.ma_state = load_state()
        self.mm_state_store = MMStateStore("state/mm_positions.json")
        
        # Initialize trade reporter
        self.trade_reporter = get_trade_reporter()
        
        # Initialize data stacks
        self.ma_data = ma_data
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
        self.last_ma_check = None
        self.clock_check_failed = False
    
    def _sync_ma_holding_from_broker(self, state):
        """Ensure state['ma_holding'] matches actual Alpaca positions."""
        ma_tickers = {
            ma_config.MA_TRADE_GROWTH,
            ma_config.MA_TRADE_SAFE,
            ma_config.MA_TRADE_ALT,
        }

        try:
            positions = broker.get_all_positions()
        except Exception as e:
            logger.error(f"Could not fetch positions to sync MA holding: {e}")
            return state.get("ma_holding"), 0.0

        active = []
        for pos in positions:
            symbol = getattr(pos, "symbol", None)
            if symbol in ma_tickers:
                qty = float(getattr(pos, "qty", 0) or 0)
                if abs(qty) > 0:
                    market_value = float(getattr(pos, "market_value", 0) or 0)
                    active.append((symbol, market_value))

        active.sort(key=lambda item: abs(item[1]), reverse=True)
        actual_symbol = active[0][0] if active else None
        actual_value = active[0][1] if active else 0.0

        if state.get("ma_holding") != actual_symbol:
            logger.warning(
                f"MA HOLDING SYNC: state={state.get('ma_holding')} -> broker={actual_symbol}"
            )
            state["ma_holding"] = actual_symbol

        state["ma_position_value"] = actual_value
        return actual_symbol, actual_value
        
    def run(self):
        """Main bot loop from 8:30 AM to 3:30 PM"""
        logger.info("=" * 60)
        logger.info("INTEGRATED BOT START" + (" [DRY RUN]" if self.dry_run else ""))
        logger.info("=" * 60)
        
        try:
            # Check if market is open
            clock = broker.get_clock()
            if not clock.is_open and not self.dry_run:
                now = market_now()
                market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
                
                # If it's after 8 AM but before market opens, idle until market open
                if now.time() >= datetime.strptime("08:00", "%H:%M").time() and now < market_open_time:
                    wait_seconds = (market_open_time - now).total_seconds()
                    logger.info(f"Market not open yet. Idling for {wait_seconds/60:.1f} minutes until market opens at 9:30 AM")
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
            
            # Exit at 3:30 PM
            if current_time >= datetime.strptime("15:30", "%H:%M").time():
                logger.info("Reached 3:30 PM - exiting")
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
            
            # Morning Momentum: 8:30 AM - 11:00 AM
            if current_time < datetime.strptime("11:00", "%H:%M").time():
                if not self.momentum_completed:
                    # If after 10:30 AM, skip morning momentum (entry window closed)
                    if current_time >= datetime.strptime("10:30", "%H:%M").time():
                        logger.info("Started after 10:30 AM - morning momentum entry window closed, skipping")
                        self.momentum_completed = True
                        continue
                    
                    # If before 9:25 AM, idle until candidate check time
                    if current_time < datetime.strptime("09:25", "%H:%M").time():
                        candidate_check_time = now.replace(hour=9, minute=25, second=0, microsecond=0)
                        wait_seconds = (candidate_check_time - now).total_seconds()
                        if wait_seconds > 0:
                            logger.info(f"Waiting for candidate check at 9:25 AM (idling {wait_seconds/60:.1f} minutes)")
                            time.sleep(wait_seconds)
                            continue  # Re-evaluate time after waking
                    
                    # Run morning momentum strategy
                    success = self.run_morning_momentum(now)
                    if success:
                        self.momentum_completed = True
                    else:
                        logger.error("Morning momentum failed to start; will retry on next loop")
                else:
                    # After momentum completes, supervise positions until 10:30
                    self._supervise_mm_positions_until_hard_exit(now)
                    continue
            
            # Check for any orphaned broker positions after emergency flatten
            self._check_orphaned_broker_positions()
            
            # 3 ETF Rotation: 11:00 AM - 3:30 PM (hourly checks)
            if current_time >= datetime.strptime("11:00", "%H:%M").time():
                self.run_etf_rotation_check(now)
                # Sleep until next hour check
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                sleep_seconds = (next_hour - now).total_seconds()
                if sleep_seconds > 0:
                    logger.info(f"Sleeping until next ETF rotation check at {next_hour.strftime('%H:%M')}")
                    time.sleep(min(sleep_seconds, 3600))  # Cap at 1 hour
    
    def _supervise_mm_positions_until_hard_exit(self, now):
        """Supervise MM positions until hard exit time, regardless of EntryLoop status."""
        hard_exit_time = datetime.strptime("10:30", "%H:%M").time()
        
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
                hard_exit_dt = now.replace(hour=10, minute=30, second=0, microsecond=0)
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
                    
                    # Get current quotes for exit price checks
                    symbols = list(current_positions.keys())
                    quotes = {}
                    for symbol in symbols:
                        try:
                            quote_dict = self.mm_data.alpaca.get_latest_quotes([symbol])
                            if quote_dict and symbol in quote_dict:
                                quotes[symbol] = quote_dict[symbol]
                        except Exception as e:
                            logger.warning(f"Failed to get quote for {symbol}: {e}")
                    
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

            # Get current prices for each tracked symbol
            price_lookup = {}
            for symbol in allowed_symbols:
                try:
                    quote_dict = self.mm_data.alpaca.get_latest_quotes([symbol])
                    quote = quote_dict.get(symbol) if quote_dict else None
                    if quote and getattr(quote, "bid_price", 0) > 0:
                        price_lookup[symbol] = quote.bid_price
                    else:
                        position = runtime_positions.get(symbol) or stored_positions.get(symbol)
                        if position:
                            price_lookup[symbol] = position.peak_price or position.entry_price
                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")
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

    def run_morning_momentum(self, now) -> bool:
        """Run morning momentum strategy"""
        logger.info("Starting morning momentum strategy")

        success = False
        try:
            # Build candidates
            candidates, stats = fetch_candidates(
                self.mm_config,
                self.mm_data,
                most_active_count=100,
                force_universe_refresh=True,
            )
            
            if not candidates:
                logger.warning("No morning momentum candidates found")
                return True
            
            # Use top candidates
            watchlist = candidates[:12]
            subscribe_symbols = [c.symbol for c in candidates[:25]]
            candidate_map = {c.symbol: c for c in candidates[:25]}
            
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
            success = True

            # Save latest state
            if self.mm_positions is not None:
                self.mm_state_store.save_positions(self.mm_positions.positions)
            self.risk_manager.persist_state()

            # Check orphaned positions post MM run
            self._check_orphaned_broker_positions()

        except Exception as e:
            logger.error(f"Error running morning momentum: {e}", exc_info=True)
            # Ensure supervision components reflect failure
            if self.mm_positions is None:
                logger.error("Morning momentum initialization failed before position manager setup")
            else:
                logger.error("Morning momentum encountered an error after initialization; supervision will attempt emergency flatten if needed")
            # On failure, avoid using partial state
            self.mm_positions = None
            self.mm_execution = None

        finally:
            # Ensure all subscriptions are cleaned up
            try:
                self.mm_data.unsubscribe_all()
            except Exception as e:
                logger.warning(f"Failed to unsubscribe data feeds: {e}")
        return success

    def run_etf_rotation_check(self, now):
        """Run 3 ETF rotation strategy check"""
        # Only check once per hour
        if self.last_ma_check and (now - self.last_ma_check).total_seconds() < 3600:
            return
            
        self.last_ma_check = now
        logger.info(f"Running 3 ETF rotation check at {now.strftime('%H:%M')}")
        
        try:
            # Sync MA holding/value from broker first
            current_ma, current_value = self._sync_ma_holding_from_broker(self.ma_state)
            logger.info(f"MA sync: holding={current_ma}, value=${current_value:,.0f}")
            pending_buy = self.ma_state.get("ma_pending_buy")
            
            # Fetch common data
            ctx = self._fetch_ma_common_data()
            if not ctx:
                return
            
            # Bootstrap MA counters on first run
            self._bootstrap_ma_counters(ctx["qqq_closes"], ctx["tlt_closes"])
            
            # Check MA crossover signal
            ma_target = ma_strategies.check_ma_crossover(
                self.ma_state, ctx["qqq_closes"], ctx["tlt_closes"]
            )
            
            # Execute MA rotation if needed or if pending buy exists
            force_pending_buy = False
            if pending_buy and pending_buy != current_ma:
                logger.warning(f"Pending MA buy detected from prior failure: {pending_buy}")
                ma_target = pending_buy
                force_pending_buy = True

            if ma_target != current_ma or force_pending_buy:
                logger.info(f"MA rotation signal: {current_ma} -> {ma_target}")
                
                # Sell current position
                if current_ma and current_ma != ma_target:
                    try:
                        position = broker.get_position(current_ma)
                        sell_qty = float(getattr(position, "qty", 0) or 0)
                    except Exception as e:
                        logger.warning(f"Could not get position quantity for {current_ma}: {e}")
                        sell_qty = 0
                    
                    if sell_qty > 0:
                        sell_price = ctx.get("live_prices", {}).get(current_ma, 0)
                        broker.close_position(current_ma)
                        
                        # Log to both systems with actual quantity
                        log_trade(self.ma_state, "SELL", current_ma, "all", sell_price, 
                                 f"MA rotation -> {ma_target}")
                        try:
                            log_trade_with_reporting(
                                current_ma,
                                "SELL",
                                sell_qty,
                                sell_price,
                                "etf_rotation",
                                notes="MA rotation",
                            )
                        except Exception:
                            logger.exception("ETF rotation reporting failed for %s SELL; continuing", current_ma)
                
                # Buy new position with same dollar value
                if ma_target:
                    # Get the value we just sold (or use target allocation)
                    current_value = self.ma_state.get("ma_position_value", 0)
                    if current_value == 0:
                        # Use 50% of equity as default allocation
                        current_value = ctx["equity"] * ma_config.MA_ALLOC_PCT
                    
                    # Update state
                    self.ma_state["ma_holding"] = ma_target
                    self.ma_state["ma_position_value"] = current_value
                    save_state(self.ma_state)
                    
                    # Get account info
                    try:
                        equity = broker.get_equity()
                        cash = broker.get_cash()
                    except Exception as e:
                        logger.error(f"Could not fetch account info: {e}")
                        equity = 100000  # Default
                        cash = equity
                    
                    invest_amount = min(current_value, cash)
                    buy_price = ctx.get("live_prices", {}).get(ma_target, 0)
                    
                    logger.info(
                        f"MA BUY: {ma_target} notional=${invest_amount:,.2f} price=${buy_price:.2f}"
                    )

                    if buy_price > 0 and invest_amount >= buy_price:
                        qty = int(invest_amount / buy_price)
                        if qty >= 1:
                            broker.buy(ma_target, qty)

                            # Log to both systems
                            log_trade(
                                self.ma_state,
                                "BUY",
                                ma_target,
                                qty,
                                buy_price,
                                f"MA rotation <- {current_ma}",
                            )
                            try:
                                log_trade_with_reporting(
                                    ma_target,
                                    "BUY",
                                    qty,
                                    buy_price,
                                    "etf_rotation",
                                    notes="MA rotation",
                                )
                            except Exception:
                                logger.exception("ETF rotation reporting failed for %s BUY; continuing", ma_target)

                            # Update state
                            self.ma_state["ma_holding"] = ma_target
                            self.ma_state["ma_position_value"] = qty * buy_price
                            self.ma_state.pop("ma_pending_buy", None)
                            save_state(self.ma_state)
                        else:
                            logger.critical(
                                f"MA rotation for {ma_target} calculated qty {qty} < 1; will retry next check"
                            )
                            self.ma_state["ma_pending_buy"] = ma_target
                            save_state(self.ma_state)
                    else:
                        logger.critical(
                            f"MA rotation buy for {ma_target} skipped (price={buy_price}, invest_amount={invest_amount})"
                        )
                        self.ma_state["ma_pending_buy"] = ma_target
                        save_state(self.ma_state)
            
            # Check for orphaned broker positions after ETF operations
            self._check_orphaned_broker_positions()
            
        except Exception as e:
            logger.error(f"Error in ETF rotation check: {e}", exc_info=True)
    
    def _fetch_ma_common_data(self):
        """Fetch common market data for MA rotation strategy."""
        try:
            equity = broker.get_equity()
            cash = broker.get_cash()
            logger.info(f"Account: equity=${equity:,.2f} cash=${cash:,.2f}")
        except Exception as e:
            logger.error(f"Could not fetch account info: {e}")
            return None
        
        try:
            logger.info("Fetching market data...")
            all_bars = ma_data.fetch_daily_bars(ma_config.ALL_TICKERS, lookback_days=150)
            
            ctx = {
                "equity": equity, "cash": cash,
                "spy_dates":  all_bars.get("SPY", {}).get("dates", []),
                "spy_closes": all_bars.get("SPY", {}).get("closes", []),
                "spy_opens":  all_bars.get("SPY", {}).get("opens", []),
                "qqq_closes": all_bars.get("QQQ", {}).get("closes", []),
                "tlt_closes": all_bars.get("TLT", {}).get("closes", []),
                "upro_closes": all_bars.get("UPRO", {}).get("closes", []),
            }
            ctx["upro_price"] = ctx["upro_closes"][-1] if ctx["upro_closes"] else 0
            
            if not ctx["spy_closes"] or not ctx["qqq_closes"] or not ctx["tlt_closes"]:
                logger.error("Missing critical price data -- aborting")
                return None
            
            # Fetch live prices
            try:
                live_tickers = ["SPY", "UPRO", ma_config.MA_TRADE_GROWTH,
                               ma_config.MA_TRADE_SAFE, ma_config.MA_TRADE_ALT]
                live = ma_data.fetch_live_prices(list(set(live_tickers)))
                ctx["spy_live"] = float(live["SPY"]) if live.get("SPY") else None
                ctx["upro_live"] = float(live["UPRO"]) if live.get("UPRO") else None
                ctx["live_prices"] = {k: float(v) for k, v in live.items() if v}
            except Exception as e:
                logger.warning(f"Live price fetch failed: {e}")
                ctx["spy_live"] = None
                ctx["upro_live"] = None
                ctx["live_prices"] = {}
            
            return ctx
        except Exception as e:
            logger.error(f"Could not fetch market data: {e}", exc_info=True)
            return None
    
    def _bootstrap_ma_counters(self, qqq_closes, tlt_closes):
        """Bootstrap MA counters on first run"""
        if self.ma_state.get("ma_bootstrapped"):
            return
        
        logger.info("First run — bootstrapping MA counters from history...")
        period = ma_config.MA_PERIOD
        buf = ma_config.MA_BUFFER_PCT
        qa, qb, ta, tb = 0, 0, 0, 0
        
        for i in range(period, len(qqq_closes)):
            qqq_sma = sum(qqq_closes[i - period + 1:i + 1]) / period
            if qqq_closes[i] > qqq_sma * (1 + buf):
                qa += 1; qb = 0
            elif qqq_closes[i] < qqq_sma * (1 - buf):
                qb += 1; qa = 0
        
        for i in range(period, len(tlt_closes)):
            tlt_sma = sum(tlt_closes[i - period + 1:i + 1]) / period
            if tlt_closes[i] > tlt_sma * (1 + buf):
                ta += 1; tb = 0
            elif tlt_closes[i] < tlt_sma * (1 - buf):
                tb += 1; ta = 0
        
        self.ma_state["ma_qa"] = qa
        self.ma_state["ma_qb"] = qb
        self.ma_state["ma_ta"] = ta
        self.ma_state["ma_tb"] = tb
        self.ma_state["ma_bootstrapped"] = True
        logger.info(f"Bootstrapped: qa={qa} qb={qb} ta={ta} tb={tb}")


def main():
    parser = argparse.ArgumentParser(description="Integrated Bot - Morning Momentum + 3 ETF Rotation")
    parser.add_argument("--dry-run", action="store_true", help="Show signals without trading")
    args = parser.parse_args()
    
    bot = IntegratedBot(dry_run=args.dry_run)
    bot.run()


if __name__ == "__main__":
    main()
