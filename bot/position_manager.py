"""Position manager for gap momentum strategy with VIX-conditioned exits"""
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
import requests
from bot import config
from bot.market_data import AlpacaDataClient

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    entry_price: float
    quantity: int
    entry_time: datetime
    entry_gap_pct: float
    adv_estimate: float
    peak_price: float = field(default=0.0)
    trailing_stop_price: float = field(default=0.0)
    is_trailing_active: bool = field(default=False)
    order_id: Optional[str] = None
    current_price: float = field(default=0.0)


@dataclass
class ExitSlicerState:
    """Tracks state for a time-based exit slicer"""
    symbol: str
    slices_remaining: int
    next_slice_time: datetime
    seconds_between_slices: int
    reason: str


class PositionManager:
    """Manages positions, entries, and VIX-conditioned exits"""

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.client = AlpacaDataClient()
        self.base_url = config.ALPACA_BASE_URL
        self.api_key = config.ALPACA_API_KEY
        self.secret_key = config.ALPACA_SECRET_KEY

        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        })

        self.exit_slicers: Dict[str, ExitSlicerState] = {}

    def load_positions(self, saved_positions: Dict):
        """Restore positions from saved state"""
        for symbol, data in saved_positions.items():
            try:
                position = Position(
                    symbol=data.get("symbol", symbol),
                    entry_price=data.get("entry_price", 0),
                    quantity=data.get("quantity", 0),
                    entry_time=datetime.fromisoformat(data.get("entry_time", datetime.now().isoformat())),
                    entry_gap_pct=data.get("entry_gap_pct", 0),
                    adv_estimate=data.get("adv_estimate", 0),
                    peak_price=data.get("peak_price", data.get("entry_price", 0)),
                    trailing_stop_price=data.get("trailing_stop_price", 0),
                    is_trailing_active=data.get("is_trailing_active", False),
                    order_id=data.get("order_id"),
                    current_price=data.get("current_price", data.get("entry_price", 0)),
                )
                self.positions[symbol] = position
                logger.info(f"Restored position: {symbol} {position.quantity} shares @ {position.entry_price:.2f}")
            except Exception as e:
                logger.error(f"Failed to restore position {symbol}: {e}")

    def get_account_equity(self) -> float:
        """Get current account equity"""
        url = f"{self.base_url}/v2/account"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return float(data.get("equity", 0))
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting account equity: {e}")
            return 0.0

    def calculate_position_size(self, target_dollars: float, adv: float, current_price: float) -> int:
        """Calculate position size with liquidity cap (0.3% of ADV)"""
        liquidity_cap = adv * config.LIQUIDITY_CAP_PCT
        position_dollars = min(target_dollars, liquidity_cap)
        quantity = int(position_dollars / current_price)
        return max(0, quantity)

    def _cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully canceled or already complete."""
        url = f"{self.base_url}/v2/orders/{order_id}"
        try:
            response = self.session.delete(url, timeout=10)
            if response.status_code in (200, 204):
                logger.info(f"Canceled order {order_id}")
                return True
            elif response.status_code == 422:
                # Order already filled or canceled
                logger.debug(f"Order {order_id} already complete (422)")
                return True
            else:
                logger.warning(f"Failed to cancel order {order_id}: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    def get_order_fill(self, order_id: str, max_wait: int = 30, allow_partial_cancel: bool = True) -> Optional[dict]:
        """
        Poll order until filled, partially filled, or timeout.
        SAFER PATTERN: On partial fill, cancel residual immediately inside this function,
        then re-read order state to return final filled quantity. This prevents the race
        window where remaining shares could fill before caller cancels.
        
        Args:
            allow_partial_cancel: If False (for MOO orders), don't aggressively cancel
                on partial fills - let the order ride to terminal state.
        """
        url = f"{self.base_url}/v2/orders/{order_id}"
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < max_wait:
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                order = response.json()

                status = order.get("status")
                filled_qty = float(order.get("filled_qty", 0))
                filled_avg_price = order.get("filled_avg_price")

                if status == "filled" and filled_qty > 0 and filled_avg_price:
                    return {"order_id": order_id, "filled_qty": filled_qty, "filled_avg_price": float(filled_avg_price), "status": "filled"}
                
                # On partial fill, optionally cancel immediately and poll until terminal state
                if status == "partially_filled" and filled_qty > 0 and filled_avg_price:
                    if not allow_partial_cancel:
                        # For MOO orders: don't cancel aggressively, just wait for terminal state
                        logger.info(f"Order {order_id} partially filled: {filled_qty} shares - waiting for terminal state (allow_partial_cancel=False)")
                        time.sleep(0.5)
                        continue
                    
                    logger.warning(f"Order {order_id} partially filled: {filled_qty} shares - canceling residual immediately")
                    cancel_success = self._cancel_order(order_id)
                    if not cancel_success:
                        logger.error(f"Failed to cancel order {order_id} - order may still be working, proceeding with caution")
                    
                    # Poll until order reaches terminal state (not just one re-read)
                    terminal_states = {"filled", "canceled", "done_for_day", "expired", "rejected"}
                    poll_start = datetime.now()
                    max_post_cancel_poll = 5  # Max 5 seconds of post-cancellation polling
                    
                    while (datetime.now() - poll_start).total_seconds() < max_post_cancel_poll:
                        time.sleep(0.5)
                        try:
                            response = self.session.get(url, timeout=10)
                            response.raise_for_status()
                            final_order = response.json()
                            final_status = final_order.get("status", "unknown")
                            
                            if final_status in terminal_states:
                                final_qty = float(final_order.get("filled_qty", filled_qty))
                                final_price = final_order.get("filled_avg_price", filled_avg_price)
                                
                                if final_qty > filled_qty:
                                    logger.info(f"Additional fill during cancellation: {final_qty - filled_qty} shares")
                                
                                return {
                                    "order_id": order_id, 
                                    "filled_qty": final_qty, 
                                    "filled_avg_price": float(final_price), 
                                    "status": "filled" if final_status == "filled" else "partially_filled"
                                }
                        except Exception as e:
                            logger.warning(f"Error polling after cancellation: {e}")
                            break
                    
                    # If we exhausted polling without terminal state, return what we have
                    logger.warning(f"Order {order_id} did not reach terminal state after cancellation - returning best known fill")
                    return {"order_id": order_id, "filled_qty": filled_qty, "filled_avg_price": float(filled_avg_price), "status": "partially_filled"}
                
                if status in ("canceled", "expired", "rejected"):
                    # Return any fills even if order was canceled/rejected
                    if filled_qty > 0 and filled_avg_price:
                        logger.warning(f"Order {order_id} {status} but {filled_qty} shares filled")
                        return {"order_id": order_id, "filled_qty": filled_qty, "filled_avg_price": float(filled_avg_price), "status": f"{status}_with_fill"}
                    logger.error(f"Order {order_id} failed with status: {status}")
                    return None

                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error polling order {order_id}: {e}")
                return None

        # Timeout check - optionally return partial fills if any, then cancel residual with polling
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            order = response.json()
            filled_qty = float(order.get("filled_qty", 0))
            filled_avg_price = order.get("filled_avg_price")
            if filled_qty > 0 and filled_avg_price:
                if not allow_partial_cancel:
                    # For MOO orders: don't cancel on timeout, just return what we have
                    logger.info(f"Order {order_id} timeout with {filled_qty} shares filled (allow_partial_cancel=False)")
                    return {"order_id": order_id, "filled_qty": filled_qty, "filled_avg_price": float(filled_avg_price), "status": "timeout_with_fill"}
                
                logger.warning(f"Order {order_id} timeout but {filled_qty} shares filled - canceling residual")
                cancel_success = self._cancel_order(order_id)
                if not cancel_success:
                    logger.error(f"Failed to cancel order {order_id} after timeout - order may still be working")
                
                # Same post-cancel polling as partial_filled branch
                terminal_states = {"filled", "canceled", "done_for_day", "expired", "rejected"}
                poll_start = datetime.now()
                max_poll = 5
                
                while (datetime.now() - poll_start).total_seconds() < max_poll:
                    time.sleep(0.5)
                    try:
                        response = self.session.get(url, timeout=10)
                        response.raise_for_status()
                        final_order = response.json()
                        final_status = final_order.get("status", "unknown")
                        
                        if final_status in terminal_states:
                            final_qty = float(final_order.get("filled_qty", filled_qty))
                            final_price = final_order.get("filled_avg_price", filled_avg_price)
                            
                            if final_qty > filled_qty:
                                logger.info(f"Additional fill during timeout cancellation: {final_qty - filled_qty} shares")
                            
                            return {
                                "order_id": order_id, 
                                "filled_qty": final_qty, 
                                "filled_avg_price": float(final_price), 
                                "status": "filled" if final_status == "filled" else "timeout_with_fill"
                            }
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Error polling after timeout cancellation: {e}")
                        break
                
                return {"order_id": order_id, "filled_qty": filled_qty, "filled_avg_price": float(filled_avg_price), "status": "timeout_with_fill"}
        except requests.exceptions.RequestException as e:
            logger.debug(f"Timeout check request failed for {order_id}: {e}")
            
        logger.warning(f"Order {order_id} did not fill within {max_wait} seconds")
        return None

    def enter_positions_moo(self, candidates: List, vix_level: float) -> List[Position]:
        """
        Enter MOO (Market On Open) orders with two-pass process:
        1. Submit all orders before 9:28 cutoff
        2. Poll fills after 9:30 auction
        """
        equity = self.get_account_equity()
        if equity <= 0:
            logger.error("Cannot enter positions: no equity")
            return []
        
        if not candidates:
            return []
        
        url = f"{self.base_url}/v2/account"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            account_data = response.json()
            cash = float(account_data.get("cash", equity))
        except requests.exceptions.RequestException:
            cash = equity
        
        # Sort candidates by ADV (lowest first) so liquidity-constrained names get allocated first
        # This allows spare funds to carry over to higher ADV candidates
        sorted_candidates = sorted(candidates, key=lambda c: c.adv_estimate)
        planned_slots = min(len(sorted_candidates), config.MAX_POSITIONS)
        selected_candidates = sorted_candidates[:planned_slots]
        
        logger.info(f"MOO Portfolio plan: {planned_slots} positions, ${cash:,.2f} cash, sorted by ADV (lowest first)")
        
        # PHASE 1: Submit all MOO orders before cutoff
        submitted = []  # List of (candidate, expected_qty, order_id, budget_used)
        remaining_cash = cash
        remaining_slots = planned_slots
        
        for i, candidate in enumerate(selected_candidates):
            symbol = candidate.symbol
            expected_price = candidate.open_price
            
            # Dynamic budget: remaining cash / remaining slots
            # This allows spare funds from liquidity-capped candidates to carry over
            per_position_budget = remaining_cash / remaining_slots if remaining_slots > 0 else 0
            
            if per_position_budget <= 0:
                logger.debug(f"Skipping {symbol}: zero budget remaining")
                continue
            
            # Calculate position size (may be capped by liquidity)
            quantity = self.calculate_position_size(per_position_budget, candidate.adv_estimate, expected_price)
            if quantity <= 0:
                logger.warning(f"Skipping {symbol}: liquidity cap prevents allocation (ADV=${candidate.adv_estimate:,.0f})")
                remaining_slots -= 1  # Still consume a slot, but cash remains available
                continue
            
            # Track actual budget used (may be less than per_position_budget due to rounding/price)
            actual_budget_used = quantity * expected_price
            
            # Submit MOO order
            order_id = self._submit_moo_order(symbol, quantity, "buy")
            if not order_id:
                remaining_slots -= 1
                continue
            
            submitted.append((candidate, quantity, order_id, actual_budget_used))
            remaining_cash -= actual_budget_used
            remaining_slots -= 1
            logger.info(f"MOO submitted [{len(submitted)}/{planned_slots}]: {symbol} {quantity} shares @ ${expected_price:.2f} (budget: ${per_position_budget:,.2f}, used: ${actual_budget_used:,.2f}, remaining: ${remaining_cash:,.2f})")
        
        logger.info(f"MOO submission phase complete: {len(submitted)}/{planned_slots} orders submitted, ${remaining_cash:,.2f} unallocated")
        
        # PHASE 2: Poll fills for all submitted orders
        entered = []
        total_allocated = 0.0
        
        for candidate, expected_qty, order_id, budget_used in submitted:
            symbol = candidate.symbol
            expected_price = candidate.open_price
            
            # MOO orders sit until 9:30 auction - use longer timeout (5 min), don't aggressively cancel
            fill = self.get_order_fill(order_id, max_wait=300, allow_partial_cancel=False)
            if not fill:
                logger.error(f"MOO fill failed for {symbol} (order_id={order_id})")
                continue
            
            actual_price = fill["filled_avg_price"]
            actual_qty = int(fill["filled_qty"])
            fill_status = fill.get("status", "unknown")
            
            if actual_qty <= 0:
                logger.error(f"No shares filled for {symbol}")
                continue
            
            actual_allocated = actual_price * actual_qty
            total_allocated += actual_allocated
            
            position = Position(
                symbol=symbol,
                entry_price=actual_price,
                quantity=actual_qty,
                entry_time=datetime.now(),
                entry_gap_pct=candidate.gap_pct,
                adv_estimate=candidate.adv_estimate,
                peak_price=actual_price,
                order_id=order_id,
            )
            self.positions[symbol] = position
            entered.append(position)
            
            if fill_status == "filled":
                # slippage = (fill_price - expected_price) / expected_price
                slippage = ((actual_price - expected_price) / expected_price) if expected_price > 0 else 0
                slippage_bps = slippage * 10000  # convert to basis points for readability
                logger.info(f"ENTER {symbol}: {actual_qty} shares @ {actual_price:.2f} (expected: {expected_price:.2f}, slippage: {slippage:+.4f} / {slippage_bps:+.0f}bps vs Massive open proxy) [VIX={vix_level:.1f}]")
            else:
                logger.warning(f"ENTER {symbol}: {actual_qty} shares @ {actual_price:.2f} (${actual_allocated:,.2f}, PARTIAL FILL - {fill_status}) [VIX={vix_level:.1f}]")
        
        remaining_cash = cash - total_allocated
        logger.info(f"MOO entry complete: {len(entered)}/{len(submitted)} filled, total=${total_allocated:,.2f}, remaining=${remaining_cash:,.2f}")
        return entered

    def _submit_market_order(self, symbol: str, quantity: int, side: str) -> Optional[str]:
        """Submit regular market order (day) for exits"""
        url = f"{self.base_url}/v2/orders"
        order_data = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        try:
            response = self.session.post(url, json=order_data, timeout=10)
            response.raise_for_status()
            data = response.json()
            order_id = data.get("id")
            logger.info(f"Market order submitted: {symbol} {side} {quantity} (ID: {order_id})")
            return order_id
        except requests.exceptions.RequestException as e:
            logger.error(f"Market order error for {symbol}: {e}")
            return None

    def _submit_moo_order(self, symbol: str, quantity: int, side: str) -> Optional[str]:
        """Submit Market On Open (MOO) order to Alpaca for auction execution"""
        url = f"{self.base_url}/v2/orders"
        order_data = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side,
            "type": "market",
            "time_in_force": "opg",  # MOO - executes at market open auction
        }
        try:
            response = self.session.post(url, json=order_data, timeout=10)
            response.raise_for_status()
            data = response.json()
            order_id = data.get("id")
            logger.info(f"MOO order submitted: {symbol} {side} {quantity} (ID: {order_id}) - will execute at 9:30 open")
            return order_id
        except requests.exceptions.RequestException as e:
            logger.error(f"MOO order error for {symbol}: {e}")
            return None

    def update_positions(self):
        """Update position state with current prices"""
        current_prices = {}
        if not self.positions:
            return current_prices

        symbols = list(self.positions.keys())
        snapshots = self.client.get_snapshots(symbols)

        for symbol, position in self.positions.items():
            snapshot = snapshots.get(symbol)
            if not snapshot:
                continue

            current_price = snapshot.get("last_price", snapshot.get("close"))
            if not current_price:
                continue

            position.current_price = current_price
            current_prices[symbol] = current_price

            if current_price > position.peak_price:
                position.peak_price = current_price

            entry_price = position.entry_price
            gain_pct = (current_price - entry_price) / entry_price

            if gain_pct >= config.TRAILING_STOP_ACTIVATION:
                position.is_trailing_active = True

            if position.is_trailing_active:
                trail_level = position.peak_price * (1 - config.TRAILING_STOP_PCT)
                position.trailing_stop_price = max(position.trailing_stop_price, trail_level)

        return current_prices

    def check_exits(self, current_time: time, vix_level: float, current_prices: Dict[str, float]) -> List[str]:
        """Check if positions should be exited based on trailing stops (full) or time (sliced)"""
        exited = []

        # Determine exit window based on VIX regime
        if vix_level < config.VIX_LOW_THRESHOLD:
            target_exit = datetime.strptime(config.EXIT_TIME_LOW_VIX, "%H:%M").time()
        elif vix_level > config.VIX_HIGH_THRESHOLD:
            target_exit = datetime.strptime(config.EXIT_TIME_HIGH_VIX, "%H:%M").time()
        else:
            target_exit = datetime.strptime(config.EXIT_TIME_MIDDLE_VIX, "%H:%M").time()

        target_dt = datetime.combine(datetime.today(), target_exit)
        exit_window_start = (target_dt - timedelta(minutes=1)).time()
        exit_window_end = (target_dt + timedelta(minutes=1)).time()

        current_dt = datetime.combine(datetime.today(), current_time)

        for symbol, position in list(self.positions.items()):
            # TRAILING STOP: Full instant exit (no slicing)
            if config.VIX_LOW_THRESHOLD <= vix_level <= config.VIX_HIGH_THRESHOLD:
                if position.is_trailing_active and position.trailing_stop_price > 0:
                    current_price = current_prices.get(symbol, position.current_price)
                    if current_price and current_price <= position.trailing_stop_price:
                        self._exit_position(symbol, "trailing_stop")
                        exited.append(symbol)
                        self.exit_slicers.pop(symbol, None)  # Cancel any active slicer
                        continue

            # TIMED EXIT: Sliced gradual exit
            in_exit_window = exit_window_start <= current_time <= exit_window_end
            has_active_slicer = symbol in self.exit_slicers

            if in_exit_window or has_active_slicer:
                # Start slicer if not already active and we're in the window
                if not has_active_slicer and in_exit_window:
                    self._start_exit_slicer(symbol, current_time, vix_level, f"time_exit (VIX={vix_level:.1f})")

                # Execute slice if it's time
                slicer = self.exit_slicers.get(symbol)
                if slicer and current_dt >= slicer.next_slice_time:
                    success = self._execute_exit_slice(symbol)
                    if success and symbol not in self.positions:
                        exited.append(symbol)

        return exited

    def _start_exit_slicer(self, symbol: str, current_time: time, vix_level: float, reason: str):
        """Start a time-based exit slicer for gradual position liquidation"""
        if symbol in self.exit_slicers:
            return

        # Determine slice parameters based on VIX regime
        if vix_level > config.VIX_HIGH_THRESHOLD:
            total_window_seconds = 6 * 60   # 2:30 regime: 6 minutes
            slices = 3
        elif vix_level < config.VIX_LOW_THRESHOLD:
            total_window_seconds = 10 * 60  # 3:30 regime: 10 minutes
            slices = 3
        else:
            total_window_seconds = 8 * 60
            slices = 3

        now_dt = datetime.combine(datetime.today(), current_time)
        self.exit_slicers[symbol] = ExitSlicerState(
            symbol=symbol,
            slices_remaining=slices,
            next_slice_time=now_dt,
            seconds_between_slices=total_window_seconds // slices,
            reason=reason,
        )
        logger.info(f"Started exit slicer for {symbol}: {slices} slices over {total_window_seconds}s ({reason})")

    def _execute_exit_slice(self, symbol: str) -> bool:
        """Execute one slice of a gradual exit. Returns True if slice executed successfully."""
        position = self.positions.get(symbol)
        slicer = self.exit_slicers.get(symbol)

        if not position or not slicer:
            self.exit_slicers.pop(symbol, None)
            return False

        # Calculate slice quantity
        if slicer.slices_remaining <= 1:
            qty = position.quantity  # Last slice: sell all remaining
        else:
            qty = max(1, position.quantity // slicer.slices_remaining)

        order_id = self._submit_market_order(symbol, qty, "sell")
        if not order_id:
            logger.error(f"Failed to submit exit slice for {symbol}")
            return False

        fill = self.get_order_fill(order_id, max_wait=30)
        if not fill:
            logger.error(f"Failed to fill exit slice for {symbol}")
            return False

        exit_price = fill["filled_avg_price"]
        filled_qty = int(fill["filled_qty"])
        fill_status = fill.get("status", "unknown")

        pnl = (exit_price - position.entry_price) * filled_qty
        pnl_pct = ((exit_price / position.entry_price) - 1) * 100

        remaining = position.quantity - filled_qty

        if remaining > 0:
            position.quantity = remaining
            slicer.slices_remaining -= 1
            slicer.next_slice_time = slicer.next_slice_time + timedelta(seconds=slicer.seconds_between_slices)
            logger.info(
                f"EXIT SLICE {symbol}: {filled_qty} @ {exit_price:.2f} "
                f"(P&L: {pnl:+.2f}, {pnl_pct:+.1f}%) - remaining {remaining}, "
                f"slices left {slicer.slices_remaining}, status={fill_status}"
            )
        else:
            logger.info(
                f"FINAL EXIT {symbol}: {filled_qty} @ {exit_price:.2f} "
                f"(P&L: {pnl:+.2f}, {pnl_pct:+.1f}%) - {slicer.reason}, status={fill_status}"
            )
            del self.positions[symbol]
            self.exit_slicers.pop(symbol, None)

        return True

    def _exit_position(self, symbol: str, reason: str):
        """
        Exit a position with market order.
        Order state finalization (cancellation, polling for terminal state) is handled 
        inside get_order_fill(). This method receives the final fill result and updates
        local position state accordingly.
        """
        position = self.positions.get(symbol)
        if not position:
            return

        order_id = self._submit_market_order(symbol, position.quantity, "sell")
        if not order_id:
            return

        fill = self.get_order_fill(order_id, max_wait=30)
        if not fill:
            logger.error(f"Failed to get exit fill for {symbol}")
            return

        exit_price = fill["filled_avg_price"]
        filled_qty = int(fill["filled_qty"])
        fill_status = fill.get("status", "unknown")
        
        pnl = (exit_price - position.entry_price) * filled_qty
        pnl_pct = ((exit_price / position.entry_price) - 1) * 100

        remaining = position.quantity - filled_qty
        
        if remaining > 0:
            position.quantity = remaining
            position.order_id = None
            logger.warning(f"PARTIAL EXIT {symbol}: {filled_qty} shares @ {exit_price:.2f} (P&L: {pnl:+.2f}) - {reason} - REMAINING: {remaining}")
        else:
            logger.info(f"EXIT {symbol}: {filled_qty} shares @ {exit_price:.2f} (P&L: {pnl:+.2f}, {pnl_pct:+.1f}%) - {reason}")
            del self.positions[symbol]

    def get_position_count(self) -> int:
        return len(self.positions)

    def force_exit_all(self, reason: str = "force"):
        logger.warning(f"Force exiting all positions: {reason}")
        for symbol in list(self.positions.keys()):
            self.exit_slicers.pop(symbol, None)  # Clear any active slicer
            self._exit_position(symbol, reason)
