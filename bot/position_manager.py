"""Position manager for gap momentum strategy with VIX-conditioned exits"""
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
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

        self.timed_exit_scheduled: Dict[str, datetime] = {}

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

    def get_order_fill(self, order_id: str, max_wait: int = 30) -> Optional[dict]:
        """
        Poll order until filled, partially filled, or timeout.
        SAFER PATTERN: On partial fill, cancel residual immediately inside this function,
        then re-read order state to return final filled quantity. This prevents the race
        window where remaining shares could fill before caller cancels.
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
                
                # SAFER PATTERN: On partial fill, cancel immediately and poll until terminal state
                if status == "partially_filled" and filled_qty > 0 and filled_avg_price:
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

        # Timeout check - return partial fills if any, then cancel residual with polling
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            order = response.json()
            filled_qty = float(order.get("filled_qty", 0))
            filled_avg_price = order.get("filled_avg_price")
            if filled_qty > 0 and filled_avg_price:
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

    def enter_positions(self, candidates: List, vix_level: float) -> List[Position]:
        """
        Enter market orders with equal capital deployment.
        CRITICAL FIX: Dynamic cash/slots recomputation after each partial fill.
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
        
        # CRITICAL FIX: Track remaining cash and slots dynamically
        remaining_cash = cash
        remaining_slots = min(len(candidates), config.MAX_POSITIONS)
        selected_candidates = candidates[:remaining_slots]
        
        logger.info(f"Portfolio plan: {remaining_slots} positions, starting with ${cash:,.2f} cash")
        
        entered = []
        total_allocated = 0.0
        
        for i, candidate in enumerate(selected_candidates):
            symbol = candidate.symbol
            expected_price = candidate.open_price
            
            # CRITICAL FIX: Recompute target per position dynamically
            slots_left = remaining_slots - len(entered)
            if slots_left <= 0:
                logger.info(f"No slots remaining, skipping remaining candidates")
                break
                
            target_per_position = remaining_cash / slots_left if slots_left > 0 else 0
            
            if target_per_position <= 0:
                logger.debug(f"Skipping {symbol}: zero target allocation")
                continue
            
            quantity = self.calculate_position_size(target_per_position, candidate.adv_estimate, expected_price)
            if quantity <= 0:
                logger.warning(f"Skipping {symbol}: liquidity cap prevents allocation")
                continue
            
            order_id = self._submit_market_order(symbol, quantity, "buy")
            if not order_id:
                continue
            
            fill = self.get_order_fill(order_id, max_wait=30)
            if not fill:
                logger.error(f"Failed to get fill for {symbol}")
                continue
            
            actual_price = fill["filled_avg_price"]
            actual_qty = int(fill["filled_qty"])
            fill_status = fill.get("status", "unknown")
            
            if actual_qty <= 0:
                logger.error(f"No shares filled for {symbol}")
                continue
            
            actual_allocated = actual_price * actual_qty
            total_allocated += actual_allocated
            remaining_cash -= actual_allocated  # CRITICAL: Update remaining cash
            
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
            
            # SAFER PATTERN: get_order_fill() already cancels residual on partial fill,
            # so we just need to log appropriately
            if fill_status == "filled":
                slippage = ((actual_price / expected_price) - 1) * 100 if expected_price > 0 else 0
                logger.info(f"ENTER {symbol}: {actual_qty} shares @ {actual_price:.2f} (${actual_allocated:,.2f}, slippage: {slippage:+.2f}%) [VIX={vix_level:.1f}]")
            else:
                logger.warning(f"ENTER {symbol}: {actual_qty} shares @ {actual_price:.2f} (${actual_allocated:,.2f}, PARTIAL FILL - {fill_status}) [VIX={vix_level:.1f}]")
            
            logger.debug(f"Remaining cash: ${remaining_cash:,.2f}, slots left: {slots_left - 1}")
        
        logger.info(f"Entry complete: {len(entered)} positions, total allocated=${total_allocated:,.2f}, remaining cash=${remaining_cash:,.2f}")
        return entered

    def _submit_market_order(self, symbol: str, quantity: int, side: str) -> Optional[str]:
        """Submit market order to Alpaca"""
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
            logger.info(f"Order submitted: {symbol} {side} {quantity} (ID: {order_id})")
            return order_id
        except requests.exceptions.RequestException as e:
            logger.error(f"Order error for {symbol}: {e}")
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
        """Check if positions should be exited based on trailing stops or time"""
        exited = []

        if vix_level < config.VIX_LOW_THRESHOLD:
            target_exit = datetime.strptime(config.EXIT_TIME_LOW_VIX, "%H:%M").time()
        elif vix_level > config.VIX_HIGH_THRESHOLD:
            target_exit = datetime.strptime(config.EXIT_TIME_HIGH_VIX, "%H:%M").time()
        else:
            target_exit = datetime.strptime(config.EXIT_TIME_MIDDLE_VIX, "%H:%M").time()

        target_dt = datetime.combine(datetime.today(), target_exit)
        exit_window_start = (target_dt - timedelta(minutes=1)).time()
        exit_window_end = (target_dt + timedelta(minutes=1)).time()

        for symbol, position in list(self.positions.items()):
            should_exit = False
            exit_reason = ""

            if config.VIX_LOW_THRESHOLD <= vix_level <= config.VIX_HIGH_THRESHOLD:
                if position.is_trailing_active and position.trailing_stop_price > 0:
                    current_price = current_prices.get(symbol, position.current_price)
                    if current_price and current_price <= position.trailing_stop_price:
                        should_exit = True
                        exit_reason = f"trailing_stop"

            if not should_exit:
                if exit_window_start <= current_time <= exit_window_end:
                    if symbol not in self.timed_exit_scheduled:
                        stagger_seconds = sum(ord(c) for c in symbol) % 120
                        target_dt = datetime.combine(datetime.today(), exit_window_start)
                        scheduled_time = target_dt + timedelta(seconds=stagger_seconds)
                        self.timed_exit_scheduled[symbol] = scheduled_time

                    scheduled = self.timed_exit_scheduled.get(symbol)
                    current_dt = datetime.combine(datetime.today(), current_time)
                    if scheduled and current_dt >= scheduled:
                        should_exit = True
                        exit_reason = f"time_exit (VIX={vix_level:.1f})"

            if should_exit:
                self._exit_position(symbol, exit_reason)
                exited.append(symbol)
                self.timed_exit_scheduled.pop(symbol, None)

        return exited

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
            self._exit_position(symbol, reason)
