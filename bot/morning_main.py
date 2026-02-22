"""Entry loop orchestration for the morning momentum bot."""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from .clock import MARKET_TZ, config_window, market_datetime, market_now
from .morning_config import Config
from .data_alpaca import Quote
from .data_sources import DataStack, init_data_stack
from .execution import ExecutionClient, ExecutionConfig, FillResult
from .indicators import VWAPState, atr_1m
from .position_manager import PositionManager, calc_qty, initial_stop_pct, _entry_client_id
from .premarket_scan import build_candidates
from .risk_manager import RiskManager
from .state_manager import StateStore
from .storage import Candidate, PendingEntryState

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    """Accumulates per-order observations for end-of-session rollup."""

    # Entry outcomes
    entry_filled: int = 0
    entry_partial: int = 0
    entry_unfilled: int = 0
    entry_unknown: int = 0

    # Exit outcomes
    exit_filled: int = 0
    exit_partial: int = 0
    exit_unfilled: int = 0
    exit_unknown: int = 0

    # Latency split by outcome type (seconds from submit to status confirmation):
    #   filled  = time to complete terminal fill
    #   partial = time to first/partial fill (IOC canceled-with-fill)
    entry_latencies_filled: List[float] = field(default_factory=list)
    entry_latencies_partial: List[float] = field(default_factory=list)
    exit_latencies_filled: List[float] = field(default_factory=list)
    exit_latencies_partial: List[float] = field(default_factory=list)

    # Slippage: fractional (fill - decision) / decision for entries,
    #           (decision - fill) / decision for exits (positive = worse)
    entry_slippages: List[float] = field(default_factory=list)
    exit_slippages: List[float] = field(default_factory=list)

    def record_entry(
        self,
        status: str,
        latency: float,
        decision_price: float,
        fill_price: float,
    ) -> None:
        if status == "filled":
            self.entry_filled += 1
            if latency > 0:
                self.entry_latencies_filled.append(latency)
        elif status == "partial":
            self.entry_partial += 1
            if latency > 0:
                self.entry_latencies_partial.append(latency)
        elif status == "unfilled":
            self.entry_unfilled += 1
        else:
            self.entry_unknown += 1
        if status in {"filled", "partial"} and decision_price > 0 and fill_price > 0:
            self.entry_slippages.append((fill_price - decision_price) / decision_price)

    def record_exit(
        self,
        status: str,
        latency: float,
        decision_price: float,
        fill_price: float,
    ) -> None:
        if status in {"filled", "dry_run"}:
            self.exit_filled += 1
            if latency > 0:
                self.exit_latencies_filled.append(latency)
        elif status == "partial":
            self.exit_partial += 1
            if latency > 0:
                self.exit_latencies_partial.append(latency)
        elif status == "unfilled":
            self.exit_unfilled += 1
        else:
            self.exit_unknown += 1
        if status in {"filled", "partial"} and decision_price > 0 and fill_price > 0:
            self.exit_slippages.append((decision_price - fill_price) / decision_price)


@dataclass
class EntryContext:
    cfg: Config
    data: DataStack
    watchlist: List[Candidate]
    max_bar_history: int
    candidate_map: Dict[str, Candidate]
    risk_manager: RiskManager
    account_equity: float
    account_cash: float
    execution: ExecutionClient
    positions: PositionManager
    state_store: StateStore
    subscribe_symbols: List[str]

    def watch_symbols(self) -> List[str]:
        return self.subscribe_symbols


@dataclass
class EntryDecision:
    symbol: str
    should_enter: bool
    reason: str
    qty: float
    entry_price: float
    stop_price: float


class EntryLoop:
    def __init__(self, ctx: EntryContext) -> None:
        self.ctx = ctx
        self.alpaca = ctx.data.alpaca
        self.bar_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=ctx.max_bar_history)
        )
        self.vwap_state: Dict[str, VWAPState] = defaultdict(VWAPState)
        self.risk_manager = ctx.risk_manager
        self.positions = ctx.positions
        self.last_prices: Dict[str, float] = {}
        self.market_open_dt = market_datetime(None, ctx.cfg.market_open)
        self.latest_quotes: Dict[str, Quote] = {}
        self.last_quote_refresh = datetime.fromtimestamp(0, tz=MARKET_TZ)
        self.stats = SessionStats()
        self.positions.stats = self.stats
        
        # Store position sizing calculated at 9:35 for consistency
        self._calculated_position_size = None
        self._positions_at_935 = 0
        
        # Track symbols that are "done for today" (partial fills, failed attempts, etc.)
        self._done_today_symbols: set[str] = set()
        
        # Track attempt counts per symbol for audit trail
        self._symbol_attempts: Dict[str, int] = {}

    def run(self) -> None:
        """Run the entry loop until entry cutoff."""
        entry_window = config_window(
            self.ctx.cfg, "entry_start", "entry_cutoff", reference=market_now()
        )
        logger.info(
            f"Entry window: {entry_window.start.strftime('%H:%M')} - {entry_window.end.strftime('%H:%M')}"
        )

        # Subscribe to live data for all watchlist symbols
        self.alpaca.subscribe_stream(self.ctx.watch_symbols())
        logger.info(f"Subscribed to {len(self.ctx.watch_symbols())} symbols")

        # Track last entry order cancellation time
        self._last_entry_cancel_check = 0.0

        try:
            while True:
                now = market_now()
                if not entry_window.contains(now):
                    logger.info("Entry window closed")
                    break

                # Process any new bars
                self._process_stream_bars()

                # Refresh quotes every 30 seconds
                if (now - self.last_quote_refresh).total_seconds() > 30:
                    self._refresh_quotes()
                    self.last_quote_refresh = now

                # Check for entries (every 5 seconds max)
                if time.monotonic() - self._last_entry_cancel_check > 5.0:
                    self._cancel_stale_entry_orders()
                    self._last_entry_cancel_check = time.monotonic()

                # Check for entries
                self._check_entries(now)

                # Check for exits
                self._check_position_exits(now)

                # Small sleep to avoid tight loop
                time.sleep(0.5)
        finally:
            # GUARANTEED: Force exit all positions at hard exit time
            self._guarantee_hard_exit()
        
        logger.info("Entry loop completed")

    def _guarantee_hard_exit(self) -> None:
        """Guarantee all positions are exited at hard exit time."""
        now = market_now()
        hard_exit_time = market_datetime(None, self.ctx.cfg.hard_exit)
        
        if now >= hard_exit_time and self.positions.positions:
            logger.warning("FORCED EXIT: Flattening all positions at hard exit time")
            
            # Step 1: Cancel any open orders first
            self._cancel_all_open_orders()
            
            # Step 2: Get current prices for all positions
            price_lookup = {}
            for symbol in self.positions.positions.keys():
                try:
                    quote_dict = self.alpaca.get_latest_quotes([symbol])
                    quote = quote_dict.get(symbol) if quote_dict else None
                    if quote and getattr(quote, "bid_price", 0) > 0:
                        price_lookup[symbol] = quote.bid_price
                    else:
                        position = self.positions.positions[symbol]
                        price_lookup[symbol] = position.peak_price or position.entry_price
                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")
                    position = self.positions.positions[symbol]
                    price_lookup[symbol] = position.peak_price or position.entry_price
            
            # Step 3: Force exit all positions
            self.positions.force_exit_all(price_lookup, reason="guaranteed_hard_exit")
            
            # Step 4: Time-based reconciliation (not dependent on stream)
            logger.info("Starting time-based exit reconciliation...")
            remaining_pending = self.positions.reconcile_pending_exits_time_based(max_wait_seconds=30.0)
            
            # Step 5: Final status
            remaining_positions = len(self.positions.positions)
            if remaining_positions > 0:
                logger.error(f"CRITICAL: {remaining_positions} positions still open after forced exit!")
                # Last resort: try to cancel any remaining orders and force close
                self._emergency_flatten()
            else:
                logger.info("All positions successfully flattened")

    def _cancel_all_open_orders(self) -> None:
        """Cancel all open orders for watchlist symbols."""
        try:
            if hasattr(self.ctx.execution, 'client') and self.ctx.execution.client:
                # Get all open orders
                orders = self.ctx.execution.client.get_orders()
                watchlist_symbols = set(self.ctx.watch_symbols())
                
                for order in orders:
                    symbol = getattr(order, 'symbol', None)
                    if symbol in watchlist_symbols:
                        order_status = getattr(order, 'status', None)
                        if order_status in {'new', 'partially_filled', 'submitted', 'accepted'}:
                            try:
                                self.ctx.execution.client.cancel_order(order.id)
                                logger.info(f"Cancelled open order {order.id} for {symbol}")
                            except Exception as e:
                                logger.warning(f"Failed to cancel order {order.id}: {e}")
        except Exception as e:
            logger.warning(f"Error cancelling open orders: {e}")

    def _emergency_flatten(self) -> None:
        """Emergency flatten: force close positions without waiting for fills."""
        logger.error("EMERGENCY FLATTEN: Force closing positions without fill confirmation")
        
        # Check broker reality before emergency flatten
        broker_positions = {}
        try:
            if hasattr(self.ctx.execution, 'client') and self.ctx.execution.client:
                positions = self.ctx.execution.client.get_positions()
                for pos in positions:
                    symbol = getattr(pos, 'symbol', None)
                    qty = float(getattr(pos, 'qty', 0) or 0)
                    if symbol and qty > 0:
                        broker_positions[symbol] = qty
                        logger.error(f"Broker reality check: {symbol} has {qty} shares")
        except Exception as e:
            logger.error(f"Could not check broker positions during emergency flatten: {e}")
        
        # Clear pending states and remove positions
        for symbol in list(self.positions.positions.keys()):
            state = self.positions.positions[symbol]
            broker_qty = broker_positions.get(symbol, 0)
            
            logger.error(f"Emergency removing position {symbol} "
                        f"(local_qty={state.qty}, broker_qty={broker_qty})")
            
            # Record as emergency exit
            state.exit_time = market_now()
            state.exit_reason = "emergency_flatten"
            state.exit_pending = False
            
            # Remove from positions
            self.positions.positions.pop(symbol, None)
        
        # Persist the cleared state
        self.positions._persist()
        
        # Log critical warning about broker reality mismatch
        if broker_positions:
            logger.critical(f"EMERGENCY FLATTEN COMPLETED WITH {len(broker_positions)} POSITIONS STILL AT BROKER")
            logger.critical("SYMBOLS REMAINING AT BROKER: " + ", ".join(f"{sym}:{qty}" for sym, qty in broker_positions.items()))
            logger.critical("MANUAL INTERVENTION REQUIRED: Check broker and close positions manually")
        else:
            logger.info("Emergency flatten completed - no positions found at broker")

    def _process_stream_bars(self) -> None:
        """Process any new bars from the stream."""
        while True:
            bar = self.alpaca.next_bar(timeout=0.1)
            if bar is None:
                break

            symbol = bar.symbol
            self.bar_history[symbol].append(bar)
            self.vwap_state[symbol].update(bar)
            self.last_prices[symbol] = bar.c

            # Update position manager with new bar
            self.positions.on_bar(symbol, bar)

    def _refresh_quotes(self) -> None:
        """Refresh latest quotes for all symbols."""
        symbols = [c.symbol for c in self.ctx.watchlist]
        quotes = self.alpaca.get_latest_quotes(symbols)
        self.latest_quotes.update(quotes)

    def _cancel_stale_entry_orders(self) -> None:
        """Cancel any remaining entry orders to prevent hanging."""
        try:
            if hasattr(self.ctx.execution, 'client') and self.ctx.execution.client:
                # Get all open orders
                orders = self.ctx.execution.client.get_orders()
                watchlist_symbols = set(self.ctx.watch_symbols())
                
                for order in orders:
                    symbol = getattr(order, 'symbol', None)
                    order_status = getattr(order, 'status', None)
                    client_order_id = getattr(order, 'client_order_id', None)
                    
                    # Cancel entry orders that are still open
                    if (symbol in watchlist_symbols and
                            client_order_id and client_order_id.startswith("ENTRY:") and
                            order_status in {'new', 'partially_filled', 'submitted', 'accepted'}):
                        
                        try:
                            self.ctx.execution.client.cancel_order(order.id)
                            logger.info(f"Cancelled stale entry order {order.id} for {symbol}")
                            
                            # Clear pending entry state
                            if client_order_id:
                                self.ctx.state_store.clear_pending_entry(client_order_id)
                                
                        except Exception as e:
                            logger.warning(f"Failed to cancel stale entry order {order.id}: {e}")
                            
        except Exception as e:
            logger.warning(f"Error cancelling stale entry orders: {e}")

    def _check_entries(self, now: datetime) -> None:
        """Check for new entry opportunities."""
        cfg = self.ctx.cfg
        
        for candidate in self.ctx.watchlist:
            symbol = candidate.symbol
            
            # Skip if we already have a position
            if self.positions.has_position(symbol):
                continue
            
            # Skip if symbol is "done for today"
            if symbol in self._done_today_symbols:
                continue

            # Check we have enough bars
            bars = list(self.bar_history.get(symbol, []))
            if len(bars) < 5:
                continue

            # Check we have RTH bars (regular trading hours)
            rth_bars = [b for b in bars if b.timestamp >= self.market_open_dt]
            if len(rth_bars) < 5:
                continue

            # First 5-minute dollar volume check
            first_5min_bars = rth_bars[:5]
            dollar_vol_5min = sum(bar.v * bar.c for bar in first_5min_bars)
            if dollar_vol_5min < cfg.min_5min_volume:
                continue

            # Opening strength check if required
            if cfg.opening_strength and first_5min_bars[-1].c <= first_5min_bars[0].o:
                continue

            # Risk check
            open_positions = self.positions.open_count
            can_enter, reason = self.risk_manager.can_enter(open_positions)
            if not can_enter:
                continue

            # Get entry price (latest quote or last bar close)
            quote = self.latest_quotes.get(symbol)
            if quote and quote.bid_price > 0 and quote.ask_price > 0:
                entry_price = (quote.bid_price + quote.ask_price) / 2
            else:
                entry_price = bars[-1].c

            # Refresh cash for each entry decision
            try:
                if hasattr(self.ctx.execution, 'client') and self.ctx.execution.client:
                    account = self.ctx.execution.client.get_account()
                    actual_cash = float(account.cash)
                else:
                    actual_cash = self.ctx.account_cash  # Fallback to snapshot
            except Exception as e:
                logger.warning(f"Could not refresh cash, using snapshot: {e}")
                actual_cash = self.ctx.account_cash
            
            # Calculate position size based on 50% of actual cash
            cash_for_positions = actual_cash * 0.50  # 50% of actual cash
            
            # Calculate position sizing once at 9:35 mark for consistency
            if self._calculated_position_size is None:
                entry_open_dt = market_datetime(None, cfg.entry_start)
                if now < entry_open_dt:
                    # Wait until entry window opens to size against stable inputs
                    continue

                max_positions = min(len(self.ctx.watchlist), cfg.max_concurrent)
                max_positions = max(max_positions, 1)

                self._positions_at_935 = max_positions
                self._calculated_position_size = cash_for_positions / max_positions

                logger.info(
                    "Position sizing anchored: $%.0f per position across %d slots",
                    self._calculated_position_size,
                    max_positions,
                )
            
            # Use the pre-calculated position size
            target_notional_per_position = self._calculated_position_size
            
            # Check daily deploy cap
            can_deploy, allowed_amount = self.risk_manager.can_deploy_amount(target_notional_per_position)
            if not can_deploy:
                logger.warning(f"Daily cap exceeded for {symbol}: target=${target_notional_per_position:.2f}, remaining=$0.00")
                continue
            
            if allowed_amount < target_notional_per_position:
                logger.info(f"Daily cap limited {symbol}: target=${target_notional_per_position:.2f}, allowed=${allowed_amount:.2f}")
            
            target_notional_per_position = allowed_amount
            
            # Calculate qty based on target notional with slippage buffer
            buy_slip_buffer = 1.0 + getattr(cfg, "exec_slippage_buy_pct", 0.0)
            effective_entry_price = entry_price * buy_slip_buffer if buy_slip_buffer > 0 else entry_price
            qty = target_notional_per_position / effective_entry_price
            
            # Apply 5% volume constraint for first 5 minutes
            if len(first_5min_bars) >= 5:
                dollar_vol_5min = sum(bar.v * bar.c for bar in first_5min_bars)
                max_notional_5pct = dollar_vol_5min * 0.05  # 5% of 5-minute volume
                max_qty_5pct = max_notional_5pct / effective_entry_price
                qty = min(qty, max_qty_5pct)
                
                logger.info("%s sizing: target=$%.0f, 5%%vol_cap=$%.0f, qty=%.0f shares", 
                           symbol, target_notional_per_position, max_notional_5pct, qty)
            
            # Check if fractional trading is allowed
            fractionable = self.ctx.execution.is_fractionable(symbol)
            
            # Only allow fractional if fractionable, otherwise floor to int
            if not fractionable:
                qty = math.floor(qty)
            
            if qty < 1:
                logger.info("%s position too small: qty=%.0f < 1", symbol, qty)
                continue

            # Calculate stop price
            atr = atr_1m(bars, 14)
            stop_pct = initial_stop_pct(cfg, atr, entry_price)
            stop_price = entry_price * (1 - stop_pct)

            # Create entry decision and execute
            decision = EntryDecision(symbol, True, "entry_signal", qty, entry_price, stop_price)
            self._execute_entry(decision)

    def _execute_entry(self, decision: EntryDecision) -> None:
        """Execute an entry order."""
        # Check if symbol already has a pending entry
        pending_entries = self.ctx.state_store.load_pending_entries()
        for client_id, pending in pending_entries.items():
            if pending.symbol == decision.symbol:
                logger.warning(f"Skipping entry for {decision.symbol} - pending entry already exists: {client_id}")
                return
        
        # Calculate attempt number from persistent tracking
        attempt = self._symbol_attempts.get(decision.symbol, 0) + 1
        self._symbol_attempts[decision.symbol] = attempt
        
        logger.info(f"Entry attempt {attempt} for {decision.symbol}")
        
        # Record pending entry BEFORE submitting
        pending_state = PendingEntryState(
            symbol=decision.symbol,
            client_order_id=_entry_client_id(decision.symbol, attempt),
            submitted_ts=time.time(),
            attempts=attempt,
            intended_qty=decision.qty,
            intended_price=decision.entry_price,
            stop_pct=(decision.entry_price - decision.stop_price) / decision.entry_price,
        )
        self.ctx.state_store.save_pending_entry(pending_state)
        
        fill = self.ctx.execution.place_entry(
            decision.symbol,
            decision.qty,
            decision.entry_price,
            client_order_id=pending_state.client_order_id,
        )

        # Record deployment amount based on actual fill, not intended amount
        if fill.status in {"filled", "partial", "dry_run"}:
            deployed_amount = float(fill.filled_qty) * float(fill.avg_price)
        else:
            deployed_amount = 0.0  # No actual deployment for unfilled/rejected orders
        self.risk_manager.on_deploy(deployed_amount)

        # Record entry stats
        latency = 0.0  # Would need to track submit time for real latency
        self.stats.record_entry(
            fill.status, latency, decision.entry_price, fill.avg_price
        )

        if fill.status in {"filled", "dry_run"}:
            # Clear pending entry on success
            self.ctx.state_store.clear_pending_entry(pending_state.client_order_id)
            
            self.positions.open_position(
                decision.symbol,
                fill.filled_qty,
                fill.avg_price,
                (decision.entry_price - decision.stop_price) / decision.entry_price,
                entry_order_id=fill.order_id,
                entry_client_order_id=pending_state.client_order_id,
            )
            logger.info(
                f"ENTRY {decision.symbol} qty={fill.filled_qty} @ {fill.avg_price:.2f} "
                f"(deployed=${deployed_amount:,.2f})"
            )
        elif fill.status == "partial":
            # Mark symbol as done for today to prevent re-entries
            logger.warning(
                f"PARTIAL ENTRY {decision.symbol} {fill.filled_qty}/{decision.qty} @ {fill.avg_price:.2f} "
                f"(deployed=${deployed_amount:,.2f}) - marking done for today"
            )
            # Clear pending entry - partial fills are considered "done for today"
            self.ctx.state_store.clear_pending_entry(pending_state.client_order_id)
            # Add to done_today_symbols to prevent re-entries
            self._done_today_symbols.add(decision.symbol)
        elif fill.status == "unfilled":
            # For IOC entries, unfilled means no liquidity - mark as done for today
            logger.info(f"IOC ENTRY {decision.symbol} unfilled - no liquidity, marking done for today (deployed=$0.00)")
            # Clear pending entry
            self.ctx.state_store.clear_pending_entry(pending_state.client_order_id)
            # Add to done_today_symbols to prevent re-entries
            self._done_today_symbols.add(decision.symbol)
        else:
            # Clear pending entry on failure
            self.ctx.state_store.clear_pending_entry(pending_state.client_order_id)
            logger.warning(f"FAILED ENTRY {decision.symbol}: {fill.status} (deployed=$0.00)")
            # Add to done_today_symbols to prevent re-entries on failed attempts
            self._done_today_symbols.add(decision.symbol)

    def _check_position_exits(self, now: datetime) -> None:
        """Check for position exits."""
        for symbol, state in list(self.positions.positions.items()):
            # Get current price
            quote = self.latest_quotes.get(symbol)
            if quote and quote.bid_price > 0:
                current_price = quote.bid_price
            else:
                bars = list(self.bar_history.get(symbol, []))
                if not bars:
                    continue
                current_price = bars[-1].c

            # Stop loss check
            if current_price <= state.stop_price:
                self.positions.exit_position(symbol, current_price, now, reason="stop_loss")
                continue

            # Hard exit time check
            hard_exit_time = market_datetime(None, self.ctx.cfg.hard_exit)
            if now >= hard_exit_time:
                self.positions.exit_position(symbol, current_price, now, reason="hard_exit")


def fetch_candidates(
    cfg: Config,
    data: DataStack,
    *,
    most_active_count: int = 50,
    force_universe_refresh: bool = False,
) -> Tuple[List[Candidate], Dict[str, any]]:
    """Fetch and filter candidates for the morning momentum strategy."""
    logger.info("Fetching morning momentum candidates...")
    
    # Get most active symbols
    symbols = data.alpaca.get_most_actives(most_active_count)
    logger.info(f"Found {len(symbols)} most active symbols")
    
    # Build candidates
    today = market_now().date()
    candidates = build_candidates(cfg, data.alpaca, data.fmp, data.float_cache, symbols, today)
    
    stats = {
        "total_symbols": len(symbols),
        "candidates_found": len(candidates),
        "scan_time": market_now(),
    }
    
    logger.info(f"Built {len(candidates)} candidates")
    return candidates, stats


def _reconcile_pending_entries(
    execution: ExecutionClient,
    state_store: StateStore,
    positions: PositionManager,
) -> None:
    """Reconcile any pending entry orders from previous session."""
    pending = state_store.load_pending_entries()
    if not pending:
        return

    logger.info(f"Reconciling {len(pending)} pending entries")
    cleared_ids = []
    unresolved = []
    for client_order_id, pending_state in pending.items():
        fill = execution.find_order_by_client_id(client_order_id)
        if fill is None:
            logger.warning(f"Pending entry {client_order_id} transient lookup failure; keeping for retry")
            unresolved.append(client_order_id)
            continue

        if fill.status in {"filled", "dry_run"}:
            positions.open_position(
                pending_state.symbol,
                fill.filled_qty,
                fill.avg_price,
                pending_state.stop_pct,
                entry_order_id=fill.order_id,
                entry_client_order_id=client_order_id,
            )
            logger.info(f"Recovered position {pending_state.symbol}")
            cleared_ids.append(client_order_id)
        elif fill.status in {"partial", "unknown"}:
            logger.warning(f"Pending entry {client_order_id} status {fill.status}; retaining for follow-up")
            unresolved.append(client_order_id)
        else:
            logger.warning(f"Pending entry {client_order_id} status: {fill.status}; clearing")
            cleared_ids.append(client_order_id)

    for client_order_id in cleared_ids:
        state_store.clear_pending_entry(client_order_id)

    if cleared_ids:
        logger.info(f"Cleared {len(cleared_ids)} reconciled pending entries")
    if unresolved:
        logger.info(f"{len(unresolved)} pending entries retained for retry")


def main() -> None:
    """Main entry point for the morning momentum bot."""
    parser = argparse.ArgumentParser(description="Morning Momentum Bot")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    args = parser.parse_args()

    # Initialize data stack
    data = init_data_stack()

    # Load configuration
    cfg = Config()

    # Initialize execution client
    exec_cfg = ExecutionConfig(
        buy_slippage_pct=cfg.exec_slippage_buy_pct,
        sell_slippage_pct=cfg.exec_slippage_sell_pct,
    )
    execution = ExecutionClient(dry_run=args.dry_run, cfg=exec_cfg)

    # Initialize state store
    state_store = StateStore("state/mm_positions.json")

    # Initialize risk manager
    risk_manager = RiskManager(cfg, state_store=state_store)

    # Initialize position manager
    positions = PositionManager(cfg, execution, risk_manager, state_store=state_store)

    # Load existing positions
    existing_positions = state_store.load_positions()
    if existing_positions:
        positions.load_states(existing_positions)

    # Reconcile pending entries
    _reconcile_pending_entries(execution, state_store, positions)

    # Fetch candidates
    candidates, stats = fetch_candidates(cfg, data, most_active_count=50)
    if not candidates:
        logger.error("No candidates found")
        return

    # Use top candidates
    watchlist = candidates[:12]
    subscribe_symbols = [c.symbol for c in candidates[:25]]
    candidate_map = {c.symbol: c for c in candidates[:25]}

    logger.info(f"Watchlist: {', '.join(c.symbol for c in watchlist)}")

    # Get account info
    try:
        account = execution.client.get_account() if not args.dry_run else None
        equity = float(account.equity) if account else 100000
        cash = float(account.cash) if account else 100000
    except Exception as e:
        logger.error(f"Could not get account info: {e}")
        equity = 100000
        cash = 100000

    # Create entry context
    ctx = EntryContext(
        cfg=cfg,
        data=data,
        watchlist=watchlist,
        max_bar_history=120,
        candidate_map=candidate_map,
        risk_manager=risk_manager,
        account_equity=equity,
        account_cash=cash,
        execution=execution,
        positions=positions,
        state_store=state_store,
        subscribe_symbols=subscribe_symbols,
    )

    # Run entry loop
    loop = EntryLoop(ctx)
    loop.run()

    # Print session stats
    stats = loop.stats
    logger.info("Session Summary:")
    logger.info(f"  Entries: {stats.entry_filled} filled, {stats.entry_partial} partial, {stats.entry_unfilled} unfilled")
    logger.info(f"  Exits: {stats.exit_filled} filled, {stats.exit_partial} partial, {stats.exit_unfilled} unfilled")


if __name__ == "__main__":
    main()
