"""Entry loop orchestration for the morning momentum bot."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional

from .clock import MARKET_TZ, config_window, market_datetime, market_now
from .morning_config import Config
from .data_alpaca import Quote
from .data_sources import DataStack
from .execution import ExecutionClient, ExecutionConfig, FillResult
from .indicators import VWAPState, atr_1m
from .monitoring import get_session_monitor
from .position_manager import PositionManager, initial_stop_pct, _entry_client_id
from .risk_manager import RiskManager
from .state_manager import StateStore
from .storage import Candidate, PendingEntryState

logger = logging.getLogger(__name__)


def allocate_positions_dynamic(
    candidates: List,
    deploy_dollars: float,
    max_per_ticker_pct: float = 0.25,
    min_order_dollars: float = 25.0,
    volume_participation_pct: float = 0.01,
) -> Dict[str, float]:
    """
    Allocate position sizes dynamically from low-volume to high-volume tickers.
    
    Strategy:
    - Sort candidates by liquidity (low → high)
    - Iterate through candidates, allocating: remaining_cash / remaining_positions
    - Apply THREE constraints: 25% cap, volume limit, and equal-weight
    - target = min(remaining/remaining_positions, 25% cap, volume_limit)
    - Stocks that hit constraints leave more cash for other positions
    - Distribute leftover to highest liquidity names
    
    Example with $10,000 deploy, 25% cap ($2,500), 1% volume participation:
    - Position 0 (liq=$50K): min($10K/10, $2.5K, $50K*0.01=$500) = $500 (VOL CAP) → $9.5K left
    - Position 1 (liq=$100K): min($9.5K/9, $2.5K, $100K*0.01=$1K) = $1K (VOL CAP) → $8.5K left
    - Position 2 (liq=$200K): min($8.5K/8, $2.5K, $200K*0.01=$2K) = $1.06K → $7.44K left
    - Position 3 (liq=$500K): min($7.44K/7, $2.5K, $500K*0.01=$5K) = $1.06K → $6.38K left
    - ...continues, unused cash from vol-capped stocks flows to others
    
    Args:
        candidates: List of Candidate objects with liq_5m_dollar set
        deploy_dollars: Total cash to deploy
        max_per_ticker_pct: Max % of deploy_dollars per ticker (default 0.25)
        min_order_dollars: Minimum order size (default $25)
        volume_participation_pct: Max % of 5-min volume per ticker (default 0.01)
    
    Returns:
        Dict[symbol, target_dollars] for each candidate
    """
    if not candidates or deploy_dollars <= 0:
        return {}
    
    # Filter tradable candidates with valid liquidity
    tradable = [c for c in candidates if c.price > 0 and c.liq_5m_dollar > 0]
    
    if not tradable:
        return {}
    
    # Sort LOW liquidity → HIGH liquidity (small names first)
    tradable.sort(key=lambda c: c.liq_5m_dollar)
    
    # Per-ticker cap in dollars
    max_per_ticker_dollars = deploy_dollars * max_per_ticker_pct
    
    # Allocate iteratively: remaining / remaining_positions, capped at max_per_ticker AND volume limit
    allocations = {}
    remaining = deploy_dollars
    
    for i, c in enumerate(tradable):
        if remaining <= min_order_dollars:
            break
        
        # Calculate how many positions are left to allocate
        remaining_positions = len(tradable) - i
        if remaining_positions <= 0:
            break
        
        # Base allocation: equal split of remaining cash
        base_allocation = remaining / remaining_positions
        
        # Volume limit: max % of this stock's 5-min dollar volume
        volume_limit = c.liq_5m_dollar * volume_participation_pct
        
        # Apply THREE constraints: equal-weight, 25% cap, volume limit
        target = min(base_allocation, max_per_ticker_dollars, volume_limit)
        
        # Enforce minimum order size
        if target < min_order_dollars:
            continue
        
        allocations[c.symbol] = target
        remaining -= target
    
    # Second pass: distribute leftover to highest liquidity names
    if remaining > min_order_dollars:
        # Sort by liquidity (high → low) for topping up
        by_high_liq = sorted(tradable, key=lambda c: c.liq_5m_dollar, reverse=True)
        
        for c in by_high_liq:
            if remaining <= min_order_dollars:
                break
            
            if c.symbol not in allocations:
                continue
            
            # Calculate room under BOTH caps (25% cap AND volume cap)
            vol_cap = c.liq_5m_dollar * volume_participation_pct
            room_cap = max_per_ticker_dollars - allocations[c.symbol]
            room_vol = vol_cap - allocations[c.symbol]
            room = min(room_cap, room_vol)
            
            if room <= 0:
                continue
            
            # Add up to room or remaining
            add = min(room, remaining)
            allocations[c.symbol] += add
            remaining -= add
    
    return allocations


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
        
        # Store dynamic position allocations calculated at entry_start
        self._position_allocations: Dict[str, float] = {}  # symbol -> target_dollars
        self._allocations_calculated = False
        
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

        # Wait for stream start time (9:28 AM) before subscribing
        from .morning_main_staged import wait_for_timeline_stage
        stream_start_time = datetime.strptime(self.ctx.cfg.stream_start, "%H:%M").time()
        if market_now().time() < stream_start_time:
            logger.info(f"Waiting for stream start at {self.ctx.cfg.stream_start}")
            wait_for_timeline_stage(self.ctx.cfg, 'stream_start')
        
        # Subscribe to live data using IEX feed
        feed = self.ctx.cfg.live_stream_feed
        logger.info(f"Subscribing to {len(self.ctx.watch_symbols())} symbols using {feed} feed")
        self.alpaca.subscribe_stream(self.ctx.watch_symbols(), feed=feed)
        logger.info(f"Subscribed to {len(self.ctx.watch_symbols())} symbols with {feed} feed")

        # Track last entry order cancellation time
        self._last_entry_cancel_check = 0.0
        self._last_pending_reconcile = 0.0

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

                # Reconcile pending entries every 10s (rate-limit safe)
                if time.monotonic() - self._last_pending_reconcile > 10.0:
                    self._reconcile_pending_entries(now)
                    self._last_pending_reconcile = time.monotonic()

                # Check for entries
                self._check_entries(now)

                # Check for exits
                self._check_position_exits(now)

                # Small sleep to avoid tight loop
                time.sleep(0.5)
        finally:
            # GUARANTEED: Force exit all positions at hard exit time
            self._guarantee_hard_exit()

            # Always release streaming subscriptions to avoid leaks
            try:
                self.alpaca.unsubscribe_all()
            except Exception as exc:
                logger.warning("Failed to unsubscribe data streams: %s", exc)
        
        logger.info("Entry loop completed")

    def _guarantee_hard_exit(self) -> None:
        """Guarantee all positions are exited at hard exit time."""
        now = market_now()
        hard_exit_time = market_datetime(None, self.ctx.cfg.hard_exit)
        
        if now >= hard_exit_time and self.positions.positions:
            logger.warning("FORCED EXIT: Flattening all positions at hard exit time")
            
            # Step 1: Cancel any open orders first
            self._cancel_all_open_orders()
            
            # Step 2: Get current prices for all positions (single batched API call)
            price_lookup = {}
            position_symbols = list(self.positions.positions.keys())
            try:
                feed = self.ctx.cfg.live_quote_refresh_feed
                quote_dict = self.alpaca.get_latest_quotes(position_symbols, feed=feed)
                for symbol in position_symbols:
                    quote = quote_dict.get(symbol) if quote_dict else None
                    if quote and getattr(quote, "bid_price", 0) > 0:
                        price_lookup[symbol] = quote.bid_price
                    else:
                        position = self.positions.positions[symbol]
                        price_lookup[symbol] = position.peak_price or position.entry_price
            except Exception as e:
                logger.warning(f"Failed to batch-fetch quotes: {e}")
                for symbol in position_symbols:
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
                    client_order_id = getattr(order, 'client_order_id', None)
                    if symbol in watchlist_symbols and client_order_id and (
                        client_order_id.startswith("ENTRY:")
                        or client_order_id.startswith("EXIT:")
                        or client_order_id.startswith("MM:")
                    ):
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
        
        client = getattr(self.ctx.execution, "client", None)

        # Update position state and request broker close
        for symbol, state in list(self.positions.positions.items()):
            broker_qty = broker_positions.get(symbol, 0)

            logger.error(
                "Emergency flattening %s (local_qty=%.4f, broker_qty=%.4f)",
                symbol,
                state.qty,
                broker_qty,
            )

            submitted_ts = time.monotonic()
            order = None
            if client is not None:
                try:
                    order = client.close_position(symbol)
                except Exception as exc:
                    logger.critical("Broker close_position failed for %s during emergency flatten: %s", symbol, exc)

            # Record emergency exit request and keep state pending for reconciliation
            state.exit_reason = "emergency_flatten"
            state.exit_pending = True
            state.exit_submitted_ts = submitted_ts
            state.exit_time = None
            if order is not None:
                state.exit_order_id = getattr(order, "id", state.exit_order_id)
                client_id = getattr(order, "client_order_id", None)
                if client_id:
                    state.exit_client_order_id = client_id

            # If broker reports zero shares, allow cleanup
            if broker_qty <= 0:
                state.exit_pending = False
                state.exit_time = market_now()
                self.positions.positions.pop(symbol, None)

        # Persist updated state after emergency actions
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
        """Refresh latest quotes for all symbols using IEX feed."""
        symbols = [c.symbol for c in self.ctx.watchlist]
        feed = self.ctx.cfg.live_quote_refresh_feed
        quotes = self.alpaca.get_latest_quotes(symbols, feed=feed)
        self.latest_quotes.update(quotes)

    def _reconcile_pending_entries(self, now: datetime) -> None:
        """Reconcile pending entries (DAY orders that returned unknown status)."""
        pending_entries = self.ctx.state_store.load_pending_entries()
        
        for client_order_id, pending in pending_entries.items():
            symbol = pending.symbol
            
            # Skip if already in positions (filled and reconciled)
            if symbol in self.positions.positions:
                self.ctx.state_store.clear_pending_entry(client_order_id)
                continue
            
            # Check order status
            fill = self.ctx.execution.find_order_by_client_id(client_order_id)
            
            if fill is None:
                # Transient error - keep pending and retry next cycle
                continue
            
            if fill.status in {"filled", "dry_run"}:
                # Order filled - open position
                logger.info(
                    f"RECONCILED ENTRY {symbol} qty={fill.filled_qty} @ {fill.avg_price:.2f}"
                )
                
                self.positions.open_position(
                    symbol,
                    fill.filled_qty,
                    fill.avg_price,
                    pending.stop_pct,
                    entry_order_id=fill.order_id,
                    entry_client_order_id=client_order_id,
                )
                
                # Record deployment
                deployed_amount = float(fill.filled_qty) * float(fill.avg_price)
                self.risk_manager.on_deploy(deployed_amount)
                
                # Record stats
                self.stats.record_entry(
                    fill.status, 0.0, pending.intended_price, fill.avg_price
                )
                
                # Clear pending entry
                self.ctx.state_store.clear_pending_entry(client_order_id)
                
            elif fill.status == "partial":
                # Partial fill - open position with partial shares
                logger.warning(
                    f"RECONCILED PARTIAL ENTRY {symbol} {fill.filled_qty} @ {fill.avg_price:.2f} "
                    f"- opening partial position and marking done"
                )

                self.positions.open_position(
                    symbol,
                    fill.filled_qty,
                    fill.avg_price,
                    pending.stop_pct,
                    entry_order_id=fill.order_id,
                    entry_client_order_id=client_order_id,
                )

                # Record deployment and stats for partial fill
                deployed_amount = float(fill.filled_qty) * float(fill.avg_price)
                self.risk_manager.on_deploy(deployed_amount)

                self.stats.record_entry(
                    fill.status, 0.0, pending.intended_price, fill.avg_price
                )

                self.ctx.state_store.clear_pending_entry(client_order_id)
                self._done_today_symbols.add(symbol)
                
            elif fill.status in {"canceled", "expired", "rejected"}:
                # Order failed - clear and mark done
                logger.info(f"RECONCILED FAILED ENTRY {symbol}: {fill.status}")
                self.ctx.state_store.clear_pending_entry(client_order_id)
                self._done_today_symbols.add(symbol)
                
            # else: status is still "unknown" - keep pending for next cycle

    def _cancel_stale_entry_orders(self) -> None:
        """Cancel resting entry orders only when stale by time or price."""
        try:
            if not (hasattr(self.ctx.execution, "client") and self.ctx.execution.client):
                return

            pending_entries = self.ctx.state_store.load_pending_entries()
            if not pending_entries:
                return

            orders = self.ctx.execution.client.get_orders()
            watchlist_symbols = set(self.ctx.watch_symbols())

            for order in orders:
                symbol = getattr(order, "symbol", None)
                order_status = str(getattr(order, "status", "")).lower()
                client_order_id = getattr(order, "client_order_id", None)

                if not (
                    symbol in watchlist_symbols
                    and client_order_id
                    and client_order_id.startswith("ENTRY:")
                    and order_status in {"new", "partially_filled", "submitted", "accepted"}
                ):
                    continue

                pending = pending_entries.get(client_order_id)
                if pending is None:
                    continue

                submitted_ts = getattr(pending, "submitted_ts", None)
                if submitted_ts is None:
                    continue

                age_seconds = time.time() - float(submitted_ts)

                # Use actual submitted order limit if available from broker
                try:
                    original_limit = float(getattr(order, "limit_price", 0.0) or 0.0)
                except Exception:
                    original_limit = 0.0

                if original_limit <= 0:
                    # Fallback to intended price if needed
                    original_limit = float(getattr(pending, "intended_price", 0.0) or 0.0)

                latest_price = None
                quote = self.latest_quotes.get(symbol)
                if quote is not None:
                    ask = getattr(quote, "ask_price", None)
                    bid = getattr(quote, "bid_price", None)
                    if ask and ask > 0:
                        latest_price = float(ask)
                    elif bid and bid > 0:
                        latest_price = float(bid)

                price_trigger = (
                    latest_price is not None
                    and original_limit > 0
                    and latest_price >= original_limit * 1.01
                )
                time_trigger = age_seconds >= 60.0

                if not (price_trigger or time_trigger):
                    continue

                try:
                    self.ctx.execution.client.cancel_order(order.id)
                    cancel_reason = "price" if price_trigger else "time"
                    logger.info(
                        "Cancelled stale entry order %s for %s (age=%.1fs, latest=%.2f, limit=%.2f, reason=%s)",
                        order.id,
                        symbol,
                        age_seconds,
                        latest_price if latest_price is not None else -1.0,
                        original_limit,
                        cancel_reason,
                    )
                    # Record canceled entry to monitoring
                    try:
                        monitor = get_session_monitor()
                        intended_qty = getattr(pending, "intended_qty", 0.0)
                        intended_price = getattr(pending, "intended_price", 0.0)
                        monitor.record_entry_order(
                            symbol=symbol,
                            intended_qty=intended_qty,
                            intended_price=intended_price,
                            submitted_limit=original_limit,
                            filled_qty=0.0,
                            avg_fill_price=0.0,
                            status="canceled",
                            cancel_reason=cancel_reason,
                            time_to_first_fill_s=age_seconds,
                        )
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"Failed to cancel stale entry order {order.id}: {e}")

        except Exception as e:
            logger.warning(f"Error cancelling stale entry orders: {e}")

    def _check_entries(self, now: datetime) -> None:
        """Check for new entry opportunities."""
        cfg = self.ctx.cfg
        
        # Calculate dynamic allocations once at entry_start (BEFORE symbol loop)
        if not self._allocations_calculated:
            entry_open_dt = market_datetime(None, cfg.entry_start)
            if now >= entry_open_dt:
                # Refresh cash
                try:
                    if hasattr(self.ctx.execution, 'client') and self.ctx.execution.client:
                        account = self.ctx.execution.client.get_account()
                        actual_cash = float(account.cash)
                    else:
                        actual_cash = self.ctx.account_cash
                except Exception as e:
                    logger.warning(f"Could not refresh cash, using snapshot: {e}")
                    actual_cash = self.ctx.account_cash
                
                # Calculate deploy amount
                deploy_dollars = actual_cash * cfg.daily_deploy_pct
                
                # Populate liquidity metrics for all candidates
                for cand in self.ctx.watchlist:
                    bars = list(self.bar_history.get(cand.symbol, []))
                    rth_bars = [b for b in bars if b.timestamp >= self.market_open_dt]
                    if len(rth_bars) >= 5:
                        first_5min_bars = rth_bars[:5]
                        cand.liq_5m_dollar = sum(bar.v * bar.c for bar in first_5min_bars)
                    else:
                        cand.liq_5m_dollar = 0.0
                
                # Calculate dynamic allocations
                self._position_allocations = allocate_positions_dynamic(
                    self.ctx.watchlist,
                    deploy_dollars,
                    max_per_ticker_pct=cfg.max_per_ticker_pct,
                    min_order_dollars=cfg.min_order_dollars,
                    volume_participation_pct=cfg.max_position_pct_of_5min_vol,
                )
                
                self._allocations_calculated = True
                
                logger.debug(
                    "Dynamic allocations calculated: %d positions, $%.0f total deploy",
                    len(self._position_allocations),
                    deploy_dollars,
                )
                
                # Log top 5 allocations
                sorted_allocs = sorted(self._position_allocations.items(), key=lambda x: x[1], reverse=True)
                for sym, amt in sorted_allocs[:5]:
                    cand = next((c for c in self.ctx.watchlist if c.symbol == sym), None)
                    liq = cand.liq_5m_dollar if cand else 0
                    logger.debug(f"  {sym}: ${amt:.0f} (liq=${liq:,.0f})")
        
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

            # Get entry price (latest quote or last bar close) - BEFORE breakout check
            quote = self.latest_quotes.get(symbol)
            if quote and quote.bid_price > 0 and quote.ask_price > 0:
                entry_price = (quote.bid_price + quote.ask_price) / 2
            else:
                entry_price = bars[-1].c

            # Opening Breakout Filter: use entry_price for consistency
            first_1min_high = rth_bars[0].h
            if cfg.opening_breakout and entry_price <= first_1min_high:
                continue

            # Risk check
            open_positions = self.positions.open_count
            can_enter, reason = self.risk_manager.can_enter(open_positions)
            if not can_enter:
                continue
            
            # Get allocated amount for this symbol
            target_notional_per_position = self._position_allocations.get(symbol, 0.0)
            if target_notional_per_position <= 0:
                # Symbol not allocated (too low volume or filtered out)
                continue
            
            # Check daily deploy cap
            can_deploy, allowed_amount = self.risk_manager.can_deploy_amount(target_notional_per_position)
            if not can_deploy:
                logger.warning(f"Daily cap exceeded for {symbol}: target=${target_notional_per_position:.2f}, remaining=$0.00")
                continue
            
            if allowed_amount < target_notional_per_position:
                logger.debug(f"Daily cap limited {symbol}: target=${target_notional_per_position:.2f}, allowed=${allowed_amount:.2f}")
            
            target_notional_per_position = allowed_amount
            
            # Calculate qty based on target notional with slippage buffer
            buy_slip_buffer = 1.0 + getattr(cfg, "exec_slippage_buy_pct", 0.0)
            effective_entry_price = entry_price * buy_slip_buffer if buy_slip_buffer > 0 else entry_price
            qty = target_notional_per_position / effective_entry_price
            
            # Apply volume constraint for first 5 minutes (use config value, default 1%)
            if len(first_5min_bars) >= 5:
                dollar_vol_5min = sum(bar.v * bar.c for bar in first_5min_bars)
                vol_cap_pct = getattr(self.ctx.cfg, "max_position_pct_of_5min_vol", 0.01)
                max_notional_vol_cap = dollar_vol_5min * vol_cap_pct
                max_qty_vol_cap = max_notional_vol_cap / effective_entry_price
                qty = min(qty, max_qty_vol_cap)
                
                logger.debug("%s sizing: target=$%.0f, %.1f%%vol_cap=$%.0f, qty=%.0f shares", 
                           symbol, target_notional_per_position, vol_cap_pct*100, max_notional_vol_cap, qty)
            
            # Check if fractional trading is allowed
            fractionable = self.ctx.execution.is_fractionable(symbol)
            
            # Only allow fractional if fractionable, otherwise floor to int
            if not fractionable:
                qty = math.floor(qty)
            
            if qty < 1:
                logger.debug("%s position too small: qty=%.0f < 1", symbol, qty)
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
        
        logger.debug(f"Entry attempt {attempt} for {decision.symbol}")
        
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
        submit_mono = time.monotonic()
        latency = submit_mono - pending_state.submitted_ts if pending_state.submitted_ts else 0.0
        self.stats.record_entry(
            fill.status, latency, decision.entry_price, fill.avg_price
        )

        # Record entry order to monitoring system
        try:
            monitor = get_session_monitor()
            is_fractional = (decision.qty % 1) != 0
            cancel_reason = ""
            if fill.status == "unfilled":
                cancel_reason = "liquidity"
            monitor.record_entry_order(
                symbol=decision.symbol,
                intended_qty=decision.qty,
                intended_price=decision.entry_price,
                submitted_limit=decision.entry_price * (1 + getattr(self.ctx.cfg, 'exec_slippage_buy_pct', 0.001)),
                filled_qty=fill.filled_qty,
                avg_fill_price=fill.avg_price,
                status=fill.status,
                signal_ts=datetime.now(MARKET_TZ).isoformat(),
                submit_ts=datetime.now(MARKET_TZ).isoformat(),
                time_to_first_fill_s=latency,
                time_to_full_fill_s=latency if fill.status == "filled" else 0.0,
                cancel_reason=cancel_reason,
                is_fractional=is_fractional,
                tif="DAY" if is_fractional else "IOC",
            )
        except Exception:
            pass

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
            # Clear pending entry and open position with partial fill
            self.ctx.state_store.clear_pending_entry(pending_state.client_order_id)

            self.positions.open_position(
                decision.symbol,
                fill.filled_qty,
                fill.avg_price,
                (decision.entry_price - decision.stop_price) / decision.entry_price,
                entry_order_id=fill.order_id,
                entry_client_order_id=pending_state.client_order_id,
            )

            self._done_today_symbols.add(decision.symbol)

            logger.warning(
                f"PARTIAL ENTRY {decision.symbol} {fill.filled_qty}/{decision.qty} @ {fill.avg_price:.2f} "
                f"(deployed=${deployed_amount:,.2f}) - opened partial position and marked done for today"
            )
        elif fill.status == "unfilled":
            # For IOC entries, unfilled means no liquidity - mark as done for today
            logger.debug(f"IOC ENTRY {decision.symbol} unfilled - no liquidity, marking done for today (deployed=$0.00)")
            # Clear pending entry
            self.ctx.state_store.clear_pending_entry(pending_state.client_order_id)
            # Add to done_today_symbols to prevent re-entries
            self._done_today_symbols.add(decision.symbol)
        elif fill.status == "unknown":
            # DAY orders (fractional shares) may return unknown - keep pending for reconciliation
            logger.debug(f"ENTRY {decision.symbol} status unknown (likely DAY order) - keeping pending for reconciliation")
            # DO NOT clear pending entry - let reconciliation handle it
            # DO NOT mark as done_today - allow reconciliation to complete
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
