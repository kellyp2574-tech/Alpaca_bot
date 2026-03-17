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
        
        # Bar arrival tracking for diagnostics
        self._bars_received_total = 0
        self._bars_received_by_symbol: Dict[str, int] = {}
        self._first_bar_symbols: set[str] = set()
        self._last_bar_summary_ts = 0.0
        
        # Entry evaluation tracking for diagnostics
        self._last_entry_diag_ts = 0.0
        self._entry_skip_reasons: Dict[str, int] = {}

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

        # ── Wait for entry window to open (accumulate bars while waiting) ──
        _wait_logged = False
        while market_now() < entry_window.start:
            if not _wait_logged:
                mins = (entry_window.start - market_now()).total_seconds() / 60.0
                logger.info(
                    "Entry window not open yet (starts %s, %.1f min away); "
                    "accumulating bars while waiting...",
                    entry_window.start.strftime("%H:%M"), mins,
                )
                _wait_logged = True
            # Keep draining the bar queue so history builds up
            self._process_stream_bars()
            time.sleep(1.0)
        
        logger.info("Entry window now open -- starting entry evaluation loop")

        try:
            while True:
                now = market_now()
                if now >= entry_window.end:
                    logger.info("Entry window closed (past %s)", entry_window.end.strftime("%H:%M"))
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

                # Reconcile pending entries every 5s (critical for DAY order fills)
                if time.monotonic() - self._last_pending_reconcile > 5.0:
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
        """Guarantee all positions are exited at hard exit time.

        After flattening local tracked positions, checks broker for any
        positions matching the MM watchlist that were never reconciled into
        bot state (e.g. DAY orders that filled but reconciliation missed).
        """
        now = market_now()
        hard_exit_time = market_datetime(None, self.ctx.cfg.hard_exit)

        if now < hard_exit_time:
            return

        # -- Step 0: Force-reconcile any pending entries before flattening --
        pending_entries = self.ctx.state_store.load_pending_entries()
        if pending_entries:
            logger.info("Hard exit: reconciling %d pending entries before flatten", len(pending_entries))
            self._reconcile_pending_entries(now)

        if self.positions.positions:
            logger.warning("FORCED EXIT: Flattening %d tracked positions at hard exit time",
                           len(self.positions.positions))

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
                self._emergency_flatten()
            else:
                logger.info("All tracked positions successfully flattened")

        # -- Step 6: Broker-reality safety net --
        # Check broker for positions matching MM watchlist that were never
        # tracked locally (the FLY scenario: filled but never reconciled).
        self._flatten_untracked_broker_positions()

    def _flatten_untracked_broker_positions(self) -> None:
        """Close any broker positions for MM watchlist symbols not tracked locally."""
        client = getattr(self.ctx.execution, "client", None)
        if client is None:
            return

        watchlist_symbols = set(self.ctx.watch_symbols())
        tracked_symbols = set(self.positions.positions.keys())

        try:
            broker_positions = client.get_all_positions()
        except Exception as e:
            logger.error("Failed to fetch broker positions for safety-net check: %s", e)
            return

        for pos in broker_positions:
            symbol = getattr(pos, "symbol", None)
            qty = float(getattr(pos, "qty", 0) or 0)
            if not symbol or qty <= 0:
                continue
            if symbol not in watchlist_symbols:
                continue
            if symbol in tracked_symbols:
                continue

            # This position exists at the broker but not in our state -- close it
            logger.error(
                "BROKER SAFETY NET: %s has %.4f shares at broker but not tracked locally -- closing",
                symbol, qty,
            )
            try:
                client.close_position(symbol)
                logger.info("BROKER SAFETY NET: close_position(%s) submitted", symbol)
            except Exception as e:
                logger.error("BROKER SAFETY NET: failed to close %s: %s", symbol, e)

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
        bars_this_call = 0
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
            
            # ── Bar arrival diagnostics ──
            bars_this_call += 1
            self._bars_received_total += 1
            self._bars_received_by_symbol[symbol] = self._bars_received_by_symbol.get(symbol, 0) + 1
            
            if symbol not in self._first_bar_symbols:
                self._first_bar_symbols.add(symbol)
                logger.info(
                    "FIRST BAR %s: c=%.2f v=%d ts=%s (symbols_with_bars=%d)",
                    symbol, bar.c, bar.v, bar.timestamp, len(self._first_bar_symbols),
                )
        
        # Periodic bar summary (every 60s)
        now_mono = time.monotonic()
        if self._bars_received_total > 0 and now_mono - self._last_bar_summary_ts >= 60.0:
            self._last_bar_summary_ts = now_mono
            syms_with_5plus = sum(1 for s, cnt in self._bars_received_by_symbol.items() if cnt >= 5)
            logger.info(
                "BAR SUMMARY: total=%d symbols_any=%d symbols_5plus=%d",
                self._bars_received_total, len(self._bars_received_by_symbol), syms_with_5plus,
            )

    def _refresh_quotes(self) -> None:
        """Refresh latest quotes for all symbols using IEX feed."""
        symbols = [c.symbol for c in self.ctx.watchlist]
        feed = self.ctx.cfg.live_quote_refresh_feed
        quotes = self.alpaca.get_latest_quotes(symbols, feed=feed)
        self.latest_quotes.update(quotes)

    # TIF-aware reconciliation timeouts (seconds)
    _IOC_UNKNOWN_TIMEOUT_S = 15.0     # IOC should resolve in <1s; 15s is very generous
    _DAY_UNKNOWN_TIMEOUT_S = 120.0    # DAY orders may sit open for a while
    _STALE_HARD_TIMEOUT_S = 300.0     # Absolute max before forced clear

    def _reconcile_pending_entries(self, now: datetime) -> None:
        """Reconcile pending entries into terminal state.

        Uses order_id-based polling as primary, falls back to client_order_id,
        and applies TIF-aware timeouts with broker position fallback.
        Unknown status CANNOT persist indefinitely.
        """
        pending_entries = self.ctx.state_store.load_pending_entries()
        if not pending_entries:
            return

        for client_order_id, pending in list(pending_entries.items()):
            symbol = pending.symbol
            age_s = time.time() - pending.submitted_ts if pending.submitted_ts > 0 else 0.0
            tif = getattr(pending, "tif", "ioc")
            unknown_timeout = (self._IOC_UNKNOWN_TIMEOUT_S if tif == "ioc"
                               else self._DAY_UNKNOWN_TIMEOUT_S)

            # Skip if already in positions (filled and reconciled elsewhere)
            if symbol in self.positions.positions:
                logger.info("RECONCILE %s: already in positions, clearing pending", symbol)
                self.ctx.state_store.clear_pending_entry(client_order_id)
                continue

            # -- Primary: poll by broker order_id (most reliable) --
            fill = None
            if pending.order_id:
                fill = self.ctx.execution.get_order_status(pending.order_id)
                if fill is None:
                    logger.warning(
                        "RECONCILE %s: order_id %s transient error (age=%.0fs)",
                        symbol, pending.order_id, age_s,
                    )

            # -- Fallback: lookup by client_order_id --
            if fill is None:
                fill = self.ctx.execution.find_order_by_client_id(client_order_id)
                if fill is None:
                    logger.warning(
                        "RECONCILE %s: client_id %s transient error (age=%.0fs)",
                        symbol, client_order_id, age_s,
                    )

            # -- Handle complete lookup failure (both returned None) --
            if fill is None:
                if age_s > unknown_timeout:
                    # Past timeout with no API response at all -- check broker
                    broker_qty = self.ctx.execution._get_broker_qty(symbol)
                    if broker_qty is not None and broker_qty > 0:
                        logger.warning(
                            "RECONCILE %s: API lookup failed but broker has %.4f shares "
                            "(age=%.0fs, tif=%s) -- adopting",
                            symbol, broker_qty, age_s, tif,
                        )
                        fill = FillResult(
                            order_id=pending.order_id,
                            filled_qty=broker_qty,
                            avg_price=pending.intended_price,
                            status="filled",
                        )
                    elif age_s > self._STALE_HARD_TIMEOUT_S:
                        logger.error(
                            "RECONCILE %s: all lookups failing for %.0fs, no broker position "
                            "-- clearing as lost (tif=%s)",
                            symbol, age_s, tif,
                        )
                        self.ctx.state_store.clear_pending_entry(client_order_id)
                        self._done_today_symbols.add(symbol)
                    # else: under hard timeout, retry next cycle
                if fill is None:
                    continue

            # -- Compute latency for monitoring --
            submitted_ts_mono = pending.submitted_ts_mono if pending.submitted_ts_mono > 0 else 0.0
            latency = time.monotonic() - submitted_ts_mono if submitted_ts_mono > 0 else 0.0

            # -- Handle terminal: filled --
            if fill.status in {"filled", "dry_run"}:
                logger.info(
                    "RECONCILED ENTRY %s qty=%.4f @ %.2f (age=%.0fs, tif=%s)",
                    symbol, fill.filled_qty, fill.avg_price, age_s, tif,
                )
                self._adopt_reconciled_entry(
                    symbol, fill, pending, client_order_id, latency,
                )
                continue

            # -- Handle terminal: partial fill --
            if fill.status == "partial":
                if fill.filled_qty > 0:
                    logger.warning(
                        "RECONCILED PARTIAL ENTRY %s qty=%.4f @ %.2f (age=%.0fs, tif=%s)",
                        symbol, fill.filled_qty, fill.avg_price, age_s, tif,
                    )
                    self._adopt_reconciled_entry(
                        symbol, fill, pending, client_order_id, latency,
                    )
                    self._done_today_symbols.add(symbol)
                else:
                    logger.info(
                        "RECONCILE %s: partial with 0 filled qty (age=%.0fs) -- clearing",
                        symbol, age_s,
                    )
                    self.ctx.state_store.clear_pending_entry(client_order_id)
                    self._done_today_symbols.add(symbol)
                continue

            # -- Handle terminal: unfilled / canceled / expired / rejected --
            if fill.status in {"unfilled", "canceled", "expired", "rejected"}:
                # Check if broker has shares anyway (race condition / status lag)
                broker_qty = None
                if age_s > 5:
                    broker_qty = self.ctx.execution._get_broker_qty(symbol)

                if broker_qty is not None and broker_qty > 0:
                    logger.warning(
                        "RECONCILE %s: order %s but broker has %.4f shares -- adopting",
                        symbol, fill.status, broker_qty,
                    )
                    adopted_fill = FillResult(
                        order_id=fill.order_id or pending.order_id,
                        filled_qty=broker_qty,
                        avg_price=pending.intended_price,
                        status="filled",
                    )
                    self._adopt_reconciled_entry(
                        symbol, adopted_fill, pending, client_order_id, latency,
                    )
                else:
                    logger.info(
                        "RECONCILE %s: order %s, no broker shares -- clearing (age=%.0fs)",
                        symbol, fill.status, age_s,
                    )
                    self.ctx.state_store.clear_pending_entry(client_order_id)
                    self._done_today_symbols.add(symbol)

                try:
                    monitor = get_session_monitor()
                    monitor.update_entry_order(
                        client_order_id=client_order_id,
                        status=fill.status,
                        time_to_first_fill_s=latency,
                    )
                except Exception:
                    pass
                continue

            # -- Handle "unknown" -- order still in-flight per broker --
            # This is where SNXX was stuck forever. Apply TIF-aware timeout.
            if age_s <= unknown_timeout:
                # Under timeout: genuinely in-flight, log and retry next cycle
                logger.info(
                    "RECONCILE %s: status=%s age=%.0fs (timeout=%.0fs, tif=%s, order_id=%s) -- waiting",
                    symbol, fill.status, age_s, unknown_timeout, tif, pending.order_id,
                )
                continue

            # Past timeout: order should have resolved by now. Check broker.
            logger.warning(
                "RECONCILE %s: status=%s PAST TIMEOUT (age=%.0fs > %.0fs, tif=%s) -- checking broker",
                symbol, fill.status, age_s, unknown_timeout, tif,
            )
            broker_qty = self.ctx.execution._get_broker_qty(symbol)

            if broker_qty is not None and broker_qty > 0:
                logger.warning(
                    "RECONCILE %s: unknown order but broker has %.4f shares -- adopting as filled",
                    symbol, broker_qty,
                )
                adopted_fill = FillResult(
                    order_id=fill.order_id or pending.order_id,
                    filled_qty=broker_qty,
                    avg_price=fill.avg_price if fill.avg_price > 0 else pending.intended_price,
                    status="filled",
                )
                self._adopt_reconciled_entry(
                    symbol, adopted_fill, pending, client_order_id, latency,
                )
            else:
                # No broker position + past timeout = order is dead
                logger.warning(
                    "RECONCILE %s: unknown order, no broker position, past %s timeout "
                    "-- clearing as expired (age=%.0fs)",
                    symbol, tif.upper(), age_s,
                )
                self.ctx.state_store.clear_pending_entry(client_order_id)
                self._done_today_symbols.add(symbol)
                try:
                    monitor = get_session_monitor()
                    monitor.update_entry_order(
                        client_order_id=client_order_id,
                        status="expired",
                        time_to_first_fill_s=latency,
                    )
                except Exception:
                    pass

    def _adopt_reconciled_entry(
        self,
        symbol: str,
        fill: "FillResult",
        pending: PendingEntryState,
        client_order_id: str,
        latency: float,
    ) -> None:
        """Open a tracked position from a reconciled fill and clear the pending entry."""
        self.positions.open_position(
            symbol,
            fill.filled_qty,
            fill.avg_price,
            pending.stop_pct,
            entry_order_id=fill.order_id,
            entry_client_order_id=client_order_id,
        )

        deployed_amount = float(fill.filled_qty) * float(fill.avg_price)
        self.risk_manager.on_deploy(deployed_amount)

        self.stats.record_entry(
            "filled", latency, pending.intended_price, fill.avg_price
        )

        try:
            monitor = get_session_monitor()
            updated = monitor.update_entry_order(
                client_order_id=client_order_id,
                filled_qty=fill.filled_qty,
                avg_fill_price=fill.avg_price,
                status="filled",
                time_to_first_fill_s=latency,
                time_to_full_fill_s=latency,
            )
            if not updated:
                # Initial record_entry_order was swallowed -- create fresh record
                logger.info(
                    "RECONCILE %s: no monitoring record for %s, creating fresh entry",
                    symbol, client_order_id,
                )
                monitor.record_entry_order(
                    symbol=symbol,
                    client_order_id=client_order_id,
                    intended_qty=pending.intended_qty,
                    intended_price=pending.intended_price,
                    submitted_limit=pending.intended_price,
                    filled_qty=fill.filled_qty,
                    avg_fill_price=fill.avg_price,
                    status="filled",
                    time_to_first_fill_s=latency,
                    time_to_full_fill_s=latency,
                )
        except Exception:
            pass

        self.ctx.state_store.clear_pending_entry(client_order_id)

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

                submitted_ts_mono = getattr(pending, "submitted_ts_mono", None)
                if submitted_ts_mono is None or submitted_ts_mono == 0.0:
                    # Fallback for old pending entries without mono timestamp
                    submitted_ts = getattr(pending, "submitted_ts", None)
                    if submitted_ts is None:
                        continue
                    # Best-effort: use wall clock (will be slightly inaccurate)
                    age_seconds = time.time() - float(submitted_ts)
                else:
                    # Correct: use monotonic clock
                    age_seconds = time.monotonic() - float(submitted_ts_mono)

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
                    # Update existing entry order record in monitoring
                    try:
                        monitor = get_session_monitor()
                        monitor.update_entry_order(
                            client_order_id=client_order_id,
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
                
                logger.info(
                    "ALLOCATIONS: %d positions sized, $%.0f total deploy, cash=$%.0f",
                    len(self._position_allocations),
                    deploy_dollars,
                    actual_cash,
                )
                
                # Log top allocations at INFO so they're always visible
                sorted_allocs = sorted(self._position_allocations.items(), key=lambda x: x[1], reverse=True)
                for sym, amt in sorted_allocs[:5]:
                    cand = next((c for c in self.ctx.watchlist if c.symbol == sym), None)
                    liq = cand.liq_5m_dollar if cand else 0
                    logger.info(f"  ALLOC {sym}: ${amt:.0f} (liq_5m=${liq:,.0f})")
                if not self._position_allocations:
                    logger.warning("ALLOCATIONS: zero positions allocated -- check liquidity/volume constraints")
        
        # ── Per-cycle skip reason tracking ──
        _skip = {"has_position": 0, "done_today": 0, "few_bars": 0, "few_rth_bars": 0,
                 "low_5m_vol": 0, "no_breakout": 0, "risk_block": 0, "no_alloc": 0,
                 "not_allocated_yet": 0, "evaluated": 0}
        
        for candidate in self.ctx.watchlist:
            symbol = candidate.symbol
            
            # Skip if we already have a position
            if self.positions.has_position(symbol):
                _skip["has_position"] += 1
                continue
            
            # Skip if symbol is "done for today"
            if symbol in self._done_today_symbols:
                _skip["done_today"] += 1
                continue

            # Check we have enough bars
            bars = list(self.bar_history.get(symbol, []))
            if len(bars) < 5:
                _skip["few_bars"] += 1
                continue

            # Check we have RTH bars (regular trading hours)
            rth_bars = [b for b in bars if b.timestamp >= self.market_open_dt]
            if len(rth_bars) < 5:
                _skip["few_rth_bars"] += 1
                continue

            # First 5-minute dollar volume check
            first_5min_bars = rth_bars[:5]
            dollar_vol_5min = sum(bar.v * bar.c for bar in first_5min_bars)
            if dollar_vol_5min < cfg.min_5min_volume:
                _skip["low_5m_vol"] += 1
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
                _skip["no_breakout"] += 1
                continue

            # Risk check
            open_positions = self.positions.open_count
            can_enter, reason = self.risk_manager.can_enter(open_positions)
            if not can_enter:
                _skip["risk_block"] += 1
                continue
            
            # Get allocated amount for this symbol
            if not self._allocations_calculated:
                _skip["not_allocated_yet"] += 1
                continue
            target_notional_per_position = self._position_allocations.get(symbol, 0.0)
            if target_notional_per_position <= 0:
                _skip["no_alloc"] += 1
                continue
            
            _skip["evaluated"] += 1
            
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

        # ── Periodic entry evaluation summary (every 60s) ──
        now_mono = time.monotonic()
        if now_mono - self._last_entry_diag_ts >= 60.0:
            self._last_entry_diag_ts = now_mono
            parts = " ".join(f"{k}={v}" for k, v in _skip.items() if v > 0)
            logger.info("ENTRY EVAL: watchlist=%d | %s", len(self.ctx.watchlist), parts or "no_symbols_checked")

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
        submit_ts_wall = time.time()
        submit_ts_mono = time.monotonic()
        
        is_fractional = (decision.qty % 1) != 0
        pending_state = PendingEntryState(
            symbol=decision.symbol,
            client_order_id=_entry_client_id(decision.symbol, attempt),
            submitted_ts=submit_ts_wall,
            submitted_ts_mono=submit_ts_mono,
            attempts=attempt,
            intended_qty=decision.qty,
            intended_price=decision.entry_price,
            stop_pct=(decision.entry_price - decision.stop_price) / decision.entry_price,
            tif="day" if is_fractional else "ioc",
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

        # Record entry stats (use monotonic clock for latency)
        now_mono = time.monotonic()
        latency = now_mono - submit_ts_mono
        self.stats.record_entry(
            fill.status, latency, decision.entry_price, fill.avg_price
        )

        # Record entry order to monitoring system (one record per client_order_id)
        try:
            monitor = get_session_monitor()
            is_fractional = (decision.qty % 1) != 0
            cancel_reason = ""
            if fill.status == "unfilled":
                cancel_reason = "liquidity"
            
            # Record initial entry order
            monitor.record_entry_order(
                symbol=decision.symbol,
                client_order_id=pending_state.client_order_id,
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
            
            # Calculate entry metadata for trade outcome recording
            cand = next((c for c in self.ctx.watchlist if c.symbol == decision.symbol), None)
            gap_at_entry = cand.gap_pct if cand else 0.0  # Store as decimal for drift classifier
            first_5min_vol = cand.liq_5m_dollar if cand else 0.0
            entry_slippage_bps = ((fill.avg_price - decision.entry_price) / decision.entry_price * 10000) if decision.entry_price > 0 else 0.0
            
            self.positions.open_position(
                decision.symbol,
                fill.filled_qty,
                fill.avg_price,
                (decision.entry_price - decision.stop_price) / decision.entry_price,
                entry_order_id=fill.order_id,
                entry_client_order_id=pending_state.client_order_id,
                gap_at_entry=gap_at_entry,
                first_5min_volume=first_5min_vol,
                fill_pct=100.0,
                entry_slippage_bps=entry_slippage_bps,
            )
            logger.info(
                f"ENTRY {decision.symbol} qty={fill.filled_qty} @ {fill.avg_price:.2f} "
                f"(deployed=${deployed_amount:,.2f})"
            )
        elif fill.status == "partial":
            # Clear pending entry and open position with partial fill
            self.ctx.state_store.clear_pending_entry(pending_state.client_order_id)

            # Calculate entry metadata for trade outcome recording
            cand = next((c for c in self.ctx.watchlist if c.symbol == decision.symbol), None)
            gap_at_entry = cand.gap_pct if cand else 0.0  # Store as decimal for drift classifier
            first_5min_vol = cand.liq_5m_dollar if cand else 0.0
            fill_pct_val = (fill.filled_qty / decision.qty * 100) if decision.qty > 0 else 0.0
            entry_slippage_bps = ((fill.avg_price - decision.entry_price) / decision.entry_price * 10000) if decision.entry_price > 0 else 0.0

            self.positions.open_position(
                decision.symbol,
                fill.filled_qty,
                fill.avg_price,
                (decision.entry_price - decision.stop_price) / decision.entry_price,
                entry_order_id=fill.order_id,
                entry_client_order_id=pending_state.client_order_id,
                gap_at_entry=gap_at_entry,
                first_5min_volume=first_5min_vol,
                fill_pct=fill_pct_val,
                entry_slippage_bps=entry_slippage_bps,
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
            # Store broker order_id for direct polling fallback
            if fill.order_id:
                pending_state.order_id = fill.order_id
                self.ctx.state_store.save_pending_entry(pending_state)
            logger.info(
                "PENDING ENTRY %s: status=unknown order_id=%s - will reconcile on next tick",
                decision.symbol, fill.order_id,
            )
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
    """Reconcile pending entry orders at startup.

    This runs once at the beginning of each session.  It must resolve every
    pending entry into a terminal state -- filled (adopt position), or
    cleared.  Nothing is "retained for follow-up" across sessions.

    Resolution priority:
      1. Purge ancient entries (prior calendar day, or no order_id + age > 60s)
      2. Poll by order_id (primary)
      3. Poll by client_order_id (fallback)
      4. Check broker position for the symbol (ultimate fallback)
      5. If still unknown, treat as lost/expired and clear
    """
    pending = state_store.load_pending_entries()
    if not pending:
        return

    now_epoch = time.time()
    today_date = datetime.now(MARKET_TZ).date()

    logger.info("STARTUP RECONCILE: %d pending entries to resolve", len(pending))
    cleared_ids = []

    for client_order_id, ps in list(pending.items()):
        symbol = ps.symbol
        age_s = now_epoch - ps.submitted_ts if ps.submitted_ts > 0 else 999999

        # ── Step 1: Purge stale entries that cannot possibly be resolved ──
        # Prior calendar day
        from datetime import date as _date_type
        entry_date = datetime.fromtimestamp(ps.submitted_ts, tz=MARKET_TZ).date() if ps.submitted_ts > 0 else None
        if entry_date is not None and entry_date < today_date:
            logger.warning(
                "STARTUP PURGE %s (%s): from prior session %s (age=%.0fs) -- clearing",
                symbol, client_order_id, entry_date.isoformat(), age_s,
            )
            cleared_ids.append(client_order_id)
            continue

        # No order_id and old enough that it will never resolve by order lookup
        if ps.order_id is None and age_s > 60:
            logger.warning(
                "STARTUP PURGE %s (%s): no order_id and age=%.0fs -- clearing",
                symbol, client_order_id, age_s,
            )
            cleared_ids.append(client_order_id)
            continue

        # ── Step 2: Try order_id lookup (most reliable) ──
        fill = None
        if ps.order_id:
            fill = execution.get_order_status(ps.order_id)
            if fill is not None:
                logger.info(
                    "STARTUP RECONCILE %s: order_id %s -> status=%s filled=%.4f",
                    symbol, ps.order_id, fill.status, fill.filled_qty,
                )

        # ── Step 3: Fallback to client_order_id lookup ──
        if fill is None:
            fill = execution.find_order_by_client_id(client_order_id)
            if fill is not None:
                logger.info(
                    "STARTUP RECONCILE %s: client_id %s -> status=%s filled=%.4f",
                    symbol, client_order_id, fill.status, fill.filled_qty,
                )
            else:
                logger.warning(
                    "STARTUP RECONCILE %s: both order lookups returned None (transient error)",
                    symbol,
                )

        # ── Step 4: Handle terminal statuses from lookup ──
        if fill is not None and fill.status in {"filled", "dry_run"}:
            if fill.filled_qty > 0 and symbol not in positions.positions:
                positions.open_position(
                    ps.symbol,
                    fill.filled_qty,
                    fill.avg_price,
                    ps.stop_pct,
                    entry_order_id=fill.order_id,
                    entry_client_order_id=client_order_id,
                )
                logger.info("STARTUP RECONCILE %s: recovered position qty=%.4f @ %.2f",
                            symbol, fill.filled_qty, fill.avg_price)
            cleared_ids.append(client_order_id)
            continue

        if fill is not None and fill.status == "partial" and fill.filled_qty > 0:
            if symbol not in positions.positions:
                positions.open_position(
                    ps.symbol,
                    fill.filled_qty,
                    fill.avg_price,
                    ps.stop_pct,
                    entry_order_id=fill.order_id,
                    entry_client_order_id=client_order_id,
                )
                logger.info("STARTUP RECONCILE %s: recovered partial position qty=%.4f @ %.2f",
                            symbol, fill.filled_qty, fill.avg_price)
            cleared_ids.append(client_order_id)
            continue

        if fill is not None and fill.status in {"unfilled", "canceled", "expired", "rejected"}:
            logger.info("STARTUP RECONCILE %s: order %s -- clearing", symbol, fill.status)
            cleared_ids.append(client_order_id)
            continue

        # ── Step 5: Unknown / lookup failed -- check broker position as last resort ──
        broker_qty = execution._get_broker_qty(symbol)
        if broker_qty is not None and broker_qty > 0:
            logger.warning(
                "STARTUP RECONCILE %s: order unresolved but broker has %.4f shares -- adopting",
                symbol, broker_qty,
            )
            if symbol not in positions.positions:
                positions.open_position(
                    ps.symbol,
                    broker_qty,
                    ps.intended_price,
                    ps.stop_pct,
                    entry_order_id=ps.order_id,
                    entry_client_order_id=client_order_id,
                )
            cleared_ids.append(client_order_id)
            continue

        # ── Step 6: No broker position, no resolvable order -- clear as lost ──
        logger.warning(
            "STARTUP RECONCILE %s: unresolvable (status=%s, broker_qty=%s, age=%.0fs) -- clearing as lost",
            symbol, fill.status if fill else "no_response", broker_qty, age_s,
        )
        cleared_ids.append(client_order_id)

    for cid in cleared_ids:
        state_store.clear_pending_entry(cid)

    logger.info("STARTUP RECONCILE: cleared %d / %d pending entries", len(cleared_ids), len(pending))
