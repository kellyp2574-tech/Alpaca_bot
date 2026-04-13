"""Position manager for gap momentum strategy with VIX-conditioned exits"""
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
import requests
from bot import config
from bot.market_data import AlpacaDataClient
from bot.rate_limiter import create_alpaca_session

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
    # Dynamic sizing: track fill quality to adapt slice sizes
    last_fill_rate: float = 1.0  # ratio of filled_qty / requested_qty (0.0-1.0)
    consecutive_partials: int = 0  # count of consecutive partial fills


@dataclass
class EntryExecutionPlan:
    """Tracks staged entry execution: open market order + post-open rescue fills"""
    symbol: str
    target_qty: int
    open_qty: int
    planned_remaining_qty: int
    expected_open_price: float
    gap_pct: float
    adv_estimate: float
    open_order_id: Optional[str] = None
    open_filled_qty: int = 0
    open_filled_avg_price: float = 0.0
    market1_order_id: Optional[str] = None
    market1_filled_qty: int = 0
    market1_filled_avg_price: float = 0.0
    market2_order_id: Optional[str] = None
    market2_filled_qty: int = 0
    market2_filled_avg_price: float = 0.0
    finalized: bool = False
    open_order_terminal: bool = False


class PositionManager:
    """Manages positions, entries, and VIX-conditioned exits"""

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.client = AlpacaDataClient()
        self.base_url = config.ALPACA_BASE_URL
        self.api_key = config.ALPACA_API_KEY
        self.secret_key = config.ALPACA_SECRET_KEY

        self.session = create_alpaca_session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        })

        self.exit_slicers: Dict[str, ExitSlicerState] = {}
        self.entry_plans: Dict[str, EntryExecutionPlan] = {}
        self.entry_stage1_done = False
        self.entry_stage2_done = False

        # Circuit breaker: track consecutive sell failures per symbol
        # Uses cooldown (not permanent ban) — after 60s the symbol can retry
        self._exit_failures: Dict[str, int] = {}
        self._exit_failure_times: Dict[str, datetime] = {}  # last failure timestamp
        self._max_exit_failures = 3  # Trigger cooldown after this many consecutive failures
        self._exit_cooldown_seconds = 60  # Seconds to wait before retrying after breaker trips

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
        """Calculate position size with ADV cap and absolute dollar cap."""
        adv_dollar_cap = adv * config.ADV_CAP_PCT
        position_dollars = min(target_dollars, adv_dollar_cap, config.MAX_POSITION_DOLLARS)
        quantity = int(position_dollars / current_price) if current_price > 0 else 0
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
                logger.info(f"Order {order_id} already complete (422)")
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
            allow_partial_cancel: If False (for open market orders), don't aggressively cancel
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
                
                # On partial fill, optionally cancel after a grace period
                if status == "partially_filled" and filled_qty > 0 and filled_avg_price:
                    if not allow_partial_cancel:
                        # For open market orders: don't cancel aggressively, just wait for terminal state
                        logger.info(f"Order {order_id} partially filled: {filled_qty} shares - waiting for terminal state (allow_partial_cancel=False)")
                        time.sleep(0.5)
                        continue
                    
                    # Wait 3 seconds before canceling to let more liquidity come in
                    logger.info(f"Order {order_id} partially filled: {filled_qty} shares - waiting 3s for more fills")
                    time.sleep(3)
                    
                    # Re-check: order might have filled during the wait
                    try:
                        recheck = self.session.get(url, timeout=10).json()
                        recheck_status = recheck.get("status")
                        recheck_qty = float(recheck.get("filled_qty", 0))
                        if recheck_status == "filled" and recheck_qty > 0:
                            return {"order_id": order_id, "filled_qty": recheck_qty, "filled_avg_price": float(recheck.get("filled_avg_price")), "status": "filled"}
                        if recheck_qty > filled_qty:
                            filled_qty = recheck_qty
                            filled_avg_price = recheck.get("filled_avg_price", filled_avg_price)
                            logger.info(f"Additional fill during grace period: now {filled_qty} shares")
                    except Exception:
                        pass
                    
                    logger.warning(f"Order {order_id} still partial after grace period ({filled_qty} shares) - canceling residual")
                    cancel_success = self._cancel_order(order_id)
                    if not cancel_success:
                        logger.error(f"Failed to cancel order {order_id} - order may still be working, proceeding with caution")
                    
                    # Poll until order reaches terminal state (10s for small-cap settle time)
                    terminal_states = {"filled", "canceled", "done_for_day", "expired", "rejected"}
                    poll_start = datetime.now()
                    max_post_cancel_poll = 10
                    
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
                    # For open market orders: don't cancel on timeout, just return what we have
                    logger.info(f"Order {order_id} timeout with {filled_qty} shares filled (allow_partial_cancel=False)")
                    return {"order_id": order_id, "filled_qty": filled_qty, "filled_avg_price": float(filled_avg_price), "status": "timeout_with_fill"}
                
                # Grace period: wait 3s before canceling to let more liquidity come in
                logger.info(f"Order {order_id} timeout with {filled_qty} shares filled - waiting 3s for more fills")
                time.sleep(3)
                
                # Re-check: order might have filled during the wait
                try:
                    recheck = self.session.get(url, timeout=10).json()
                    recheck_status = recheck.get("status")
                    recheck_qty = float(recheck.get("filled_qty", 0))
                    if recheck_status == "filled" and recheck_qty > 0:
                        return {"order_id": order_id, "filled_qty": recheck_qty, "filled_avg_price": float(recheck.get("filled_avg_price")), "status": "filled"}
                    if recheck_qty > filled_qty:
                        filled_qty = recheck_qty
                        filled_avg_price = recheck.get("filled_avg_price", filled_avg_price)
                        logger.info(f"Additional fill during timeout grace period: now {filled_qty} shares")
                except Exception:
                    pass
                
                logger.warning(f"Order {order_id} still partial after timeout grace period ({filled_qty} shares) - canceling residual")
                cancel_success = self._cancel_order(order_id)
                if not cancel_success:
                    logger.error(f"Failed to cancel order {order_id} after timeout - order may still be working")
                
                # Post-cancel polling (10s for small-cap settle time)
                terminal_states = {"filled", "canceled", "done_for_day", "expired", "rejected"}
                poll_start = datetime.now()
                max_poll = 10
                
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

    def enter_positions_moo(self, candidates: List, vix_level: float, capital_override: Optional[float] = None) -> Tuple[List[Position], float]:
        """LEGACY — MOO entry from gap strategy. Not used by overnight momentum."""
        raise NotImplementedError("enter_positions_moo is legacy gap-strategy code; use submit_buy_order instead")

    def _submit_market_order(self, symbol: str, quantity: int, side: str) -> Optional[str]:
        """Submit regular market order (day). Used for buy-side entry orders only."""
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
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            body = ""
            try:
                body = e.response.text[:500] if e.response is not None else ""
            except Exception:
                body = "<unreadable>"
            logger.error(
                f"Market order FAILED | symbol={symbol} | side={side} | qty={quantity} | "
                f"url={url} | status={status_code} | body={body}"
            )
            return None
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Market order FAILED (network) | symbol={symbol} | side={side} | "
                f"qty={quantity} | url={url} | error={e}"
            )
            return None


    def _submit_sell_order(self, symbol: str, qty: int, order_type: str = "market",
                           limit_price: Optional[float] = None) -> Optional[dict]:
        """Submit a sell order via POST /v2/orders.
        
        ALL exits go through this method — never DELETE /v2/positions.
        POST /v2/orders is more reliable and avoids Alpaca's extra restrictions
        on the close-position endpoint.
        
        Args:
            symbol: The symbol to sell.
            qty: Number of shares to sell.
            order_type: 'market' or 'limit'.
            limit_price: Required if order_type='limit'.
        
        Returns:
            Order response dict (with 'id') or None on failure.
        """
        url = f"{self.base_url}/v2/orders"
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "sell",
            "type": order_type,
            "time_in_force": "day",
        }
        if order_type == "limit" and limit_price is not None:
            order_data["limit_price"] = f"{limit_price:.4f}"

        try:
            response = self.session.post(url, json=order_data, timeout=10)
            response.raise_for_status()
            data = response.json()
            order_id = data.get("id")
            price_info = f" @ {limit_price:.4f}" if limit_price else ""
            logger.info(f"Sell order submitted: {symbol} {order_type} {qty}{price_info} (ID: {order_id})")
            # Reset failure counter and timestamp on success
            self._exit_failures.pop(symbol, None)
            self._exit_failure_times.pop(symbol, None)
            return data
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            body = ""
            try:
                body = e.response.text[:500] if e.response is not None else ""
            except Exception:
                body = "<unreadable>"
            logger.error(
                f"Sell order FAILED | symbol={symbol} | type={order_type} | qty={qty} | "
                f"url={url} | status={status_code} | body={body}"
            )
            self._exit_failures[symbol] = self._exit_failures.get(symbol, 0) + 1
            self._exit_failure_times[symbol] = datetime.now()
            if self._exit_failures[symbol] >= self._max_exit_failures:
                logger.error(
                    f"CIRCUIT BREAKER: {symbol} has failed {self._exit_failures[symbol]} "
                    f"consecutive close attempts — {self._exit_cooldown_seconds}s cooldown"
                )
            return None
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Sell order FAILED (network) | symbol={symbol} | type={order_type} | "
                f"qty={qty} | url={url} | error={e}"
            )
            self._exit_failures[symbol] = self._exit_failures.get(symbol, 0) + 1
            self._exit_failure_times[symbol] = datetime.now()
            if self._exit_failures[symbol] >= self._max_exit_failures:
                logger.error(
                    f"CIRCUIT BREAKER: {symbol} has failed {self._exit_failures[symbol]} "
                    f"consecutive close attempts — {self._exit_cooldown_seconds}s cooldown"
                )
            return None

    def submit_buy_order(self, symbol: str, qty: int) -> Optional[dict]:
        """Submit a market buy order via POST /v2/orders.

        Used for overnight momentum entries at 3:50 PM.
        Returns the order response dict or None on failure.
        """
        url = f"{self.base_url}/v2/orders"
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        }

        try:
            response = self.session.post(url, json=order_data, timeout=10)
            response.raise_for_status()
            data = response.json()
            order_id = data.get("id")
            logger.info(f"Buy order submitted: {symbol} x{qty} (ID: {order_id})")
            return data
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            body = ""
            try:
                body = e.response.text[:500] if e.response is not None else ""
            except Exception:
                body = "<unreadable>"
            logger.error(
                f"Buy order FAILED | symbol={symbol} | qty={qty} | "
                f"status={status_code} | body={body}"
            )
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Buy order FAILED (network) | symbol={symbol} | qty={qty} | error={e}")
            return None

    def _get_last_price(self, symbol: str) -> Optional[float]:
        """Get last trade price for a symbol (for limit sell fallback)."""
        try:
            snapshots = self.client.get_snapshots([symbol])
            snap = snapshots.get(symbol, {}) if snapshots else {}
            return snap.get("last_price") or snap.get("close")
        except Exception:
            return None

    def _is_exit_blocked(self, symbol: str) -> bool:
        """Check if a symbol is in cooldown after hitting the circuit breaker.
        
        Returns False (not blocked) if:
        - symbol has fewer than _max_exit_failures consecutive failures, OR
        - the cooldown period has elapsed (resets the counter to allow retry)
        """
        failures = self._exit_failures.get(symbol, 0)
        if failures < self._max_exit_failures:
            return False
        # Breaker tripped — check if cooldown has elapsed
        last_fail = self._exit_failure_times.get(symbol)
        if last_fail and (datetime.now() - last_fail).total_seconds() >= self._exit_cooldown_seconds:
            # Cooldown elapsed: reset counter, allow retry
            logger.info(f"Circuit breaker cooldown elapsed for {symbol} — allowing retry")
            self._exit_failures[symbol] = 0
            self._exit_failure_times.pop(symbol, None)
            return False
        remaining = self._exit_cooldown_seconds - (datetime.now() - last_fail).total_seconds() if last_fail else 0
        logger.debug(f"Circuit breaker active for {symbol}: {failures} failures, {remaining:.0f}s cooldown remaining")
        return True

    def _submit_marketable_limit_order(self, symbol: str, quantity: int, side: str, limit_price: float) -> Optional[str]:
        """Submit aggressive day limit order for post-open entry."""
        url = f"{self.base_url}/v2/orders"
        order_data = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side,
            "type": "limit",
            "time_in_force": "day",
            "limit_price": f"{limit_price:.4f}",
        }
        try:
            response = self.session.post(url, json=order_data, timeout=10)
            response.raise_for_status()
            data = response.json()
            order_id = data.get("id")
            logger.info(f"Marketable limit order submitted: {symbol} {side} {quantity} @ {limit_price:.4f} (ID: {order_id})")
            return order_id
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            body = ""
            try:
                body = e.response.text[:500] if e.response is not None else ""
            except Exception:
                body = "<unreadable>"
            logger.error(
                f"Marketable limit order FAILED | symbol={symbol} | side={side} | "
                f"qty={quantity} | limit={limit_price:.4f} | url={url} | "
                f"status={status_code} | body={body}"
            )
            return None
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Marketable limit order FAILED (network) | symbol={symbol} | side={side} | "
                f"qty={quantity} | limit={limit_price:.4f} | url={url} | error={e}"
            )
            return None

    def _get_aggressive_buy_limit(self, symbol: str, fallback_open: float) -> Optional[float]:
        """
        Build an aggressive buy limit from live quote/snapshot.
        Uses ask if available, otherwise last price, with 50 bps buffer.
        """
        snapshots = self.client.get_snapshots([symbol])
        snapshot = snapshots.get(symbol, {}) if snapshots else {}
        return self._get_aggressive_buy_limit_from_snapshot(snapshot, fallback_open)

    def _get_aggressive_buy_limit_from_snapshot(self, snapshot: dict, fallback_open: float) -> Optional[float]:
        """LEGACY — gap-strategy aggressive buy limit. Not used by overnight momentum."""
        raise NotImplementedError("_get_aggressive_buy_limit_from_snapshot is legacy code")

    def build_entry_plans(self, candidates: List, capital_override: Optional[float] = None) -> Dict[str, EntryExecutionPlan]:
        """LEGACY — staged gap-strategy entry plans. Not used by overnight momentum."""
        raise NotImplementedError("build_entry_plans is legacy code; overnight strategy sizes in momentum_scorer.allocate_head_tail")

    def submit_open_entry_orders(self, plans: Dict[str, EntryExecutionPlan], state_saver=None) -> None:
        """LEGACY — gap-strategy open entry orders. Not used by overnight momentum."""
        raise NotImplementedError("submit_open_entry_orders is legacy code")

    def reconcile_open_order_fills(self) -> None:
        """Poll market order status for up to 90s. Only records fills for terminal orders.
        
        Non-terminal orders after timeout are left pending (open_order_terminal=False)
        for the broker repair sync at 9:35 to resolve. This prevents declaring
        still-working orders as no-fills.
        """
        active_orders = {}
        for symbol, plan in self.entry_plans.items():
            if not plan.open_order_id or plan.open_order_terminal:
                continue
            active_orders[plan.open_order_id] = {"symbol": symbol, "plan": plan}

        if not active_orders:
            return

        logger.info(f"Reconciling {len(active_orders)} market orders (batch polling, 90s window)...")
        terminal_states = {"filled", "canceled", "done_for_day", "expired", "rejected"}
        start_time = datetime.now()
        max_wait = 90  # penny stock market orders can take 3-4 min to fill at open

        while active_orders and (datetime.now() - start_time).total_seconds() < max_wait:
            completed = []

            for order_id, meta in list(active_orders.items()):
                url = f"{self.base_url}/v2/orders/{order_id}"
                try:
                    response = self.session.get(url, timeout=5)
                    response.raise_for_status()
                    order = response.json()
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Open order poll error {order_id}: {e}")
                    continue

                status = order.get("status")
                filled_qty = int(float(order.get("filled_qty", 0)))
                filled_avg_price = order.get("filled_avg_price")

                if status in terminal_states:
                    plan = meta["plan"]
                    symbol = meta["symbol"]
                    plan.open_filled_qty = filled_qty
                    plan.open_filled_avg_price = float(filled_avg_price) if filled_avg_price else 0.0
                    plan.open_order_terminal = True

                    if plan.open_filled_avg_price > 0:
                        plan.expected_open_price = plan.open_filled_avg_price
                        logger.info(f"Market order TERMINAL: {symbol} filled {filled_qty}/{plan.open_qty} @ {plan.expected_open_price:.4f} (status={status})")
                    elif filled_qty > 0:
                        logger.info(f"Market order TERMINAL: {symbol} filled {filled_qty}/{plan.open_qty} (status={status})")
                    else:
                        logger.info(f"Market order TERMINAL: {symbol} no fill (status={status})")
                    completed.append(order_id)
                else:
                    # Log partial progress for non-terminal orders
                    if filled_qty > 0:
                        logger.info(f"Market order PENDING: {symbol} partial {filled_qty}/{meta['plan'].open_qty} (status={status})")

            for order_id in completed:
                active_orders.pop(order_id, None)

            if active_orders:
                time.sleep(1.0)

        # Log remaining non-terminal orders — these will be resolved by broker_repair_sync at 9:35
        if active_orders:
            pending_symbols = [meta["symbol"] for meta in active_orders.values()]
            logger.warning(
                f"Reconciliation timeout: {len(active_orders)} orders still non-terminal after {max_wait}s — "
                f"will resolve at broker repair sync. Symbols: {pending_symbols}"
            )

    def broker_repair_sync(self) -> List[str]:
        """Delayed broker repair sync — resolves pending entry orders against broker ground truth.
        
        Called repeatedly from Phase 3 (9:35–9:38). For each unresolved plan:
        1. Fetch all broker positions (single API call).
        2. If broker holds the symbol: import qty + avg price, mark terminal.
        3. If broker doesn't hold it: check order status.
           - Terminal with fills → record, mark terminal.
           - Terminal with no fills → confirmed no-fill, mark terminal.
           - Still non-terminal → leave unresolved for retry on next call.
        
        Does NOT force-mark non-terminal orders as no-fill. The caller
        (Phase 3 in _step3_manage_staged_entry) handles the hard cutoff
        at 9:38 via finalize_entry_positions(force=True).
        
        Returns list of symbols that were repaired from broker data.
        """
        # Identify unresolved plans (submitted but not yet terminal)
        unresolved = {}
        for symbol, plan in self.entry_plans.items():
            if plan.open_order_id and not plan.open_order_terminal and not plan.finalized:
                unresolved[symbol] = plan

        if not unresolved:
            logger.info("Broker repair sync: all orders already terminal — nothing to resolve")
            return []

        logger.info(f"Broker repair sync: resolving {len(unresolved)} pending orders")

        # Fetch broker positions (ground truth)
        try:
            broker_positions = self.get_broker_positions()
            broker_map = {
                pos.get("symbol"): pos for pos in broker_positions
            } if broker_positions else {}
        except Exception as e:
            logger.error(f"Broker repair sync: failed to fetch positions: {e}")
            broker_map = {}

        repaired = []
        terminal_states = {"filled", "canceled", "done_for_day", "expired", "rejected"}

        for symbol, plan in unresolved.items():
            broker_pos = broker_map.get(symbol)

            if broker_pos:
                # Broker holds this symbol — import position data
                broker_qty = abs(int(float(broker_pos.get("qty", 0))))
                broker_avg = float(broker_pos.get("avg_entry_price", 0) or 0)

                plan.open_filled_qty = broker_qty
                plan.open_filled_avg_price = broker_avg
                if broker_avg > 0:
                    plan.expected_open_price = broker_avg
                plan.open_order_terminal = True

                logger.warning(
                    f"Broker repair: {symbol} filled {broker_qty} @ {broker_avg:.4f} "
                    f"(recovered from broker positions)"
                )
                repaired.append(symbol)
            else:
                # Broker doesn't hold it — one final order status check
                try:
                    url = f"{self.base_url}/v2/orders/{plan.open_order_id}"
                    response = self.session.get(url, timeout=5)
                    response.raise_for_status()
                    order = response.json()
                    status = order.get("status")
                    filled_qty = int(float(order.get("filled_qty", 0)))
                    filled_avg_price = order.get("filled_avg_price")

                    if status in terminal_states:
                        plan.open_filled_qty = filled_qty
                        plan.open_filled_avg_price = float(filled_avg_price) if filled_avg_price else 0.0
                        plan.open_order_terminal = True
                        if filled_qty > 0:
                            if plan.open_filled_avg_price > 0:
                                plan.expected_open_price = plan.open_filled_avg_price
                            logger.warning(f"Broker repair: {symbol} late fill {filled_qty} (status={status})")
                            repaired.append(symbol)
                        else:
                            logger.info(f"Broker repair: {symbol} confirmed no fill (status={status})")
                    else:
                        # Still non-terminal — leave unresolved for retry on next call.
                        # Don't force no-fill yet; Alpaca may still be working the order.
                        logger.warning(
                            f"Broker repair: {symbol} still non-terminal (status={status}) "
                            f"and broker has no position — leaving unresolved for retry"
                        )
                except requests.exceptions.RequestException as e:
                    # Can't reach order — leave unresolved rather than guessing
                    logger.error(
                        f"Broker repair: {symbol} order check failed ({e}) "
                        f"— leaving unresolved for retry"
                    )

        still_unresolved = sum(
            1 for s, p in unresolved.items()
            if not p.open_order_terminal
        )
        logger.info(
            f"Broker repair sync complete: {len(repaired)} repaired, "
            f"{still_unresolved} still unresolved"
        )
        return repaired

    def refresh_no_fill_prices(self) -> None:
        """
        Refresh expected_open_price for symbols with no market order fill using live post-open snapshot.
        This ensures rescue pass chase guards use true post-open prices, not stale pre-open proxies.
        Covers ALL non-finalized plans with open_filled_qty == 0, including plans where
        open_qty was 0 (tiny positions that skipped market order submission entirely).
        """
        symbols_to_refresh = []
        for symbol, plan in self.entry_plans.items():
            if plan.finalized or plan.symbol in self.positions:
                continue
            if plan.open_filled_qty == 0:
                symbols_to_refresh.append(symbol)
        
        if not symbols_to_refresh:
            return
        
        logger.info(f"Refreshing prices for {len(symbols_to_refresh)} no-fill symbols before rescue pass")
        snapshots = self.client.get_snapshots(symbols_to_refresh)
        
        for symbol in symbols_to_refresh:
            plan = self.entry_plans[symbol]
            snapshot = snapshots.get(symbol, {}) if snapshots else {}
            
            # Use last trade price or daily bar open as best proxy for true open
            true_open = snapshot.get("open") or snapshot.get("last_price") or snapshot.get("close")
            if true_open and true_open > 0:
                old_price = plan.expected_open_price
                plan.expected_open_price = true_open
                logger.info(f"Price refresh: {symbol} expected_open_price {old_price:.4f} -> {true_open:.4f} (no market order fill)")

    def submit_post_open_rescue_pass(self, pass_name: str = "market1", state_saver=None) -> None:
        """LEGACY — gap-strategy post-open rescue pass. Not used by overnight momentum."""
        raise NotImplementedError("submit_post_open_rescue_pass is legacy code")

    def finalize_entry_positions(self, force: bool = False) -> List[Position]:
        """Create final Position objects from confirmed entry fills.
        
        Args:
            force: If True, finalize ALL plans including non-terminal orders
                   (used at 9:35 after broker_repair_sync). If False, only
                   finalize plans where the order is confirmed terminal.
        """
        entered = []

        for symbol, plan in self.entry_plans.items():
            if plan.finalized:
                continue

            total_qty = plan.open_filled_qty + plan.market1_filled_qty + plan.market2_filled_qty
            if total_qty <= 0:
                if force or plan.open_order_terminal:
                    plan.finalized = True
                    logger.info(f"Entry plan finalized (no fill): {symbol}")
                # else: order still working, skip — will resolve at broker repair sync
                continue

            total_cost = (
                plan.open_filled_qty * plan.open_filled_avg_price +
                plan.market1_filled_qty * plan.market1_filled_avg_price +
                plan.market2_filled_qty * plan.market2_filled_avg_price
            )
            avg_price = total_cost / total_qty

            position = Position(
                symbol=symbol,
                entry_price=avg_price,
                quantity=total_qty,
                entry_time=datetime.now(),
                entry_gap_pct=plan.gap_pct,
                adv_estimate=plan.adv_estimate,
                peak_price=avg_price,
                order_id=plan.market2_order_id or plan.market1_order_id or plan.open_order_id,
            )
            self.positions[symbol] = position
            entered.append(position)
            plan.finalized = True

            logger.info(f"FINAL ENTRY {symbol}: {total_qty} shares @ {avg_price:.4f}")

        return entered

    def update_positions(self):
        """LEGACY — VIX-based trailing stop position updater. Not used by overnight momentum."""
        raise NotImplementedError("update_positions is legacy VIX code; overnight strategy manages exits in integrated_main")

    def check_exits(self, current_time: dt_time, vix_level: float, current_prices: Dict[str, float]) -> List[str]:
        """LEGACY — VIX-conditioned exit logic. Not used by overnight momentum."""
        raise NotImplementedError("check_exits is legacy VIX code; overnight strategy uses _check_hard_stops/_check_drop_stops/_exit_all_positions")

    def _start_exit_slicer(self, symbol: str, current_time: dt_time, vix_level: float, reason: str):
        """LEGACY — VIX-conditioned exit slicer. Not used by overnight momentum."""
        raise NotImplementedError("_start_exit_slicer is legacy VIX code")

    def _execute_exit_slice(self, symbol: str) -> bool:
        """Execute one slice of a gradual exit. Returns True if slice executed successfully.
        
        On partial fill: immediately resubmits for remaining slice qty (no waiting).
        On market sell failure: falls back to aggressive limit sell.
        """
        position = self.positions.get(symbol)
        slicer = self.exit_slicers.get(symbol)

        if not position or not slicer:
            self.exit_slicers.pop(symbol, None)
            return False

        # Circuit breaker: stop retrying if we've hit max failures
        if self._is_exit_blocked(symbol):
            return False

        # Calculate slice quantity (liquidity-aware + dynamic)
        if slicer.slices_remaining <= 1:
            qty = position.quantity  # Last slice: sell all remaining
        else:
            base_qty = max(1, position.quantity // slicer.slices_remaining)
            qty = base_qty

            # Dynamic sizing: reduce slice if previous fills were poor
            if slicer.last_fill_rate < 1.0 and slicer.last_fill_rate > 0:
                adjusted = max(1, int(qty * slicer.last_fill_rate))
                if adjusted < qty:
                    logger.info(f"Dynamic sizing {symbol}: reducing slice {qty} → {adjusted} (fill rate {slicer.last_fill_rate:.0%})")
                    qty = adjusted
            # After 2+ consecutive partials, halve aggressively
            if slicer.consecutive_partials >= 2:
                halved = max(1, qty // 2)
                if halved < qty:
                    logger.warning(f"Dynamic sizing {symbol}: halving slice {qty} → {halved} ({slicer.consecutive_partials} consecutive partials)")
                    qty = halved

            # Floor: never shrink below 25% of base slice (prevents infinite ratchet-down)
            min_qty = max(1, base_qty // 4)
            if qty < min_qty:
                logger.warning(f"Dynamic sizing {symbol}: clamping {qty} → {min_qty} (25% floor of base {base_qty})")
                qty = min_qty

        # Cap slice size to 1% of ADV to avoid overwhelming thin books
        adv = getattr(position, 'adv_estimate', 0) or 0
        if adv > 0 and position.current_price and position.current_price > 0:
            adv_shares = adv / position.current_price
            max_slice = max(1, int(adv_shares * 0.01))
            if qty > max_slice and slicer.slices_remaining > 1:
                qty = max_slice
                # Recalculate slices needed for the remainder
                new_slices = max(1, (position.quantity + max_slice - 1) // max_slice)
                if new_slices > slicer.slices_remaining:
                    slicer.slices_remaining = new_slices
                    logger.info(f"Liquidity cap: {symbol} slice capped to {qty} shares (1% ADV), slices expanded to {new_slices}")

        # Submit sell order via POST /v2/orders
        sell_resp = self._submit_sell_order(symbol, qty)

        # Fallback: if market sell fails, try aggressive limit sell
        if not sell_resp:
            last_price = self._get_last_price(symbol)
            if last_price and last_price > 0:
                limit_price = round(last_price * 0.97, 4)  # 3% below last trade
                logger.warning(f"Market sell failed for {symbol}, trying limit sell @ {limit_price:.4f}")
                sell_resp = self._submit_sell_order(symbol, qty, order_type="limit", limit_price=limit_price)

        if not sell_resp:
            logger.error(f"Failed to submit exit slice for {symbol} (market + limit both failed)")
            return False

        order_id = sell_resp.get("id")
        if not order_id:
            logger.error(f"Sell order response missing order ID for {symbol}")
            return False

        fill = self.get_order_fill(order_id, max_wait=30)
        if not fill:
            logger.error(f"Failed to fill exit slice for {symbol}")
            return False

        exit_price = fill["filled_avg_price"]
        filled_qty = int(fill["filled_qty"])
        fill_status = fill.get("status", "unknown")

        # Update dynamic fill tracking for next slice sizing
        slicer.last_fill_rate = filled_qty / qty if qty > 0 else 1.0
        if filled_qty < qty:
            slicer.consecutive_partials += 1
        else:
            slicer.consecutive_partials = 0

        pnl = (exit_price - position.entry_price) * filled_qty
        pnl_pct = ((exit_price / position.entry_price) - 1) * 100

        remaining = position.quantity - filled_qty

        if remaining > 0:
            position.quantity = remaining

            # If partial fill left residual from THIS slice, resubmit immediately
            slice_residual = qty - filled_qty
            if slice_residual > 0 and fill_status != "filled":
                logger.warning(
                    f"EXIT SLICE {symbol}: partial fill {filled_qty}/{qty} — "
                    f"resubmitting {slice_residual} immediately"
                )
                resub_resp = self._submit_sell_order(symbol, slice_residual)
                if resub_resp:
                    resub_id = resub_resp.get("id")
                    if resub_id:
                        resub_fill = self.get_order_fill(resub_id, max_wait=15)
                        if resub_fill:
                            resub_qty = int(resub_fill["filled_qty"])
                            remaining -= resub_qty
                            position.quantity = remaining
                            logger.info(f"Resubmit filled {resub_qty} more for {symbol}, remaining={remaining}")

            slicer.slices_remaining -= 1
            slicer.next_slice_time = slicer.next_slice_time + timedelta(seconds=slicer.seconds_between_slices)
            logger.info(
                f"EXIT SLICE {symbol}: {filled_qty} @ {exit_price:.2f} "
                f"(P&L: {pnl:+.2f}, {pnl_pct:+.1f}%) - remaining {remaining}, "
                f"slices left {slicer.slices_remaining}, fill_rate={slicer.last_fill_rate:.0%}, status={fill_status}"
            )

            # If resubmit cleaned up the rest
            if remaining <= 0:
                logger.info(f"FINAL EXIT {symbol} (resubmit completed): {slicer.reason}")
                self.positions.pop(symbol, None)
                self.exit_slicers.pop(symbol, None)
        else:
            logger.info(
                f"FINAL EXIT {symbol}: {filled_qty} @ {exit_price:.2f} "
                f"(P&L: {pnl:+.2f}, {pnl_pct:+.1f}%) - {slicer.reason}, status={fill_status}"
            )
            del self.positions[symbol]
            self.exit_slicers.pop(symbol, None)

        return True

    def _exit_position(self, symbol: str, reason: str):
        """Exit a full position via POST /v2/orders (market sell with limit fallback).
        
        On partial fill: immediately resubmits for remaining qty.
        On market sell failure: falls back to aggressive limit sell.
        """
        position = self.positions.get(symbol)
        if not position:
            return

        # Circuit breaker: stop retrying if we've hit max failures
        if self._is_exit_blocked(symbol):
            return

        qty = position.quantity
        sell_resp = self._submit_sell_order(symbol, qty)

        # Fallback: if market sell fails, try aggressive limit sell
        if not sell_resp:
            last_price = self._get_last_price(symbol)
            if last_price and last_price > 0:
                limit_price = round(last_price * 0.97, 4)
                logger.warning(f"Market sell failed for {symbol}, trying limit sell @ {limit_price:.4f}")
                sell_resp = self._submit_sell_order(symbol, qty, order_type="limit", limit_price=limit_price)

        if not sell_resp:
            logger.error(f"Failed to exit {symbol} (market + limit both failed) - {reason}")
            return

        order_id = sell_resp.get("id")
        if not order_id:
            logger.error(f"Sell order response missing order ID for {symbol}")
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
            logger.warning(
                f"PARTIAL EXIT {symbol}: {filled_qty}/{qty} @ {exit_price:.2f} "
                f"(P&L: {pnl:+.2f}) - {reason} - resubmitting {remaining}"
            )
            # Immediate resubmit for residual
            resub_resp = self._submit_sell_order(symbol, remaining)
            if resub_resp:
                resub_id = resub_resp.get("id")
                if resub_id:
                    resub_fill = self.get_order_fill(resub_id, max_wait=15)
                    if resub_fill:
                        resub_qty = int(resub_fill["filled_qty"])
                        remaining -= resub_qty
                        position.quantity = remaining
                        logger.info(f"Resubmit filled {resub_qty} more for {symbol}, remaining={remaining}")
            if remaining <= 0:
                logger.info(f"EXIT {symbol} (resubmit completed): {reason}")
                self.positions.pop(symbol, None)
        else:
            logger.info(f"EXIT {symbol}: {filled_qty} shares @ {exit_price:.2f} (P&L: {pnl:+.2f}, {pnl_pct:+.1f}%) - {reason}")
            del self.positions[symbol]

    def get_position_count(self) -> int:
        return len(self.positions)

    def get_total_capital(self) -> float:
        """Get total buying power from account for deployment tracking"""
        url = f"{self.base_url}/v2/account"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            account_data = response.json()
            return float(account_data.get("buying_power", 0))
        except requests.exceptions.RequestException:
            return self.get_account_equity()

    def force_exit_all(self, reason: str = "force"):
        logger.warning(f"Force exiting all positions: {reason}")
        for symbol in list(self.positions.keys()):
            self.exit_slicers.pop(symbol, None)  # Clear any active slicer
            self._exit_position(symbol, reason)

    def cancel_all_open_orders(self) -> bool:
        """Cancel all open Alpaca orders."""
        url = f"{self.base_url}/v2/orders"
        try:
            response = self.session.delete(url, timeout=15)
            if response.status_code in (200, 204):
                logger.warning("Canceled all open orders")
                return True
            logger.warning(f"Cancel all orders returned HTTP {response.status_code}: {response.text}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error canceling all open orders: {e}")
            return False

    def get_open_orders(self) -> List[dict]:
        """Get all open orders from Alpaca."""
        url = f"{self.base_url}/v2/orders"
        params = {"status": "open"}
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    def cancel_open_buy_orders(self) -> int:
        """Cancel all open market buy orders (used for startup cleanup). Returns count canceled."""
        open_orders = self.get_open_orders()
        # Look for market orders with day time_in_force (our entry orders)
        market_buy_orders = [
            order for order in open_orders
            if order.get("type") == "market" and order.get("side") == "buy" and order.get("time_in_force") == "day"
        ]
        
        if not market_buy_orders:
            return 0
        
        logger.warning(f"Found {len(market_buy_orders)} open market buy orders on startup - canceling to prevent duplicates")
        
        canceled_count = 0
        for order in market_buy_orders:
            order_id = order.get("id")
            symbol = order.get("symbol")
            qty = order.get("qty")
            
            if self._cancel_order(order_id):
                logger.warning(f"Canceled orphaned market order: {symbol} {qty} shares (ID: {order_id})")
                canceled_count += 1
            else:
                logger.error(f"Failed to cancel market order: {symbol} {qty} shares (ID: {order_id})")
        
        return canceled_count

    def cancel_orphaned_open_orders(self) -> int:
        """
        Cancel only truly orphaned market buy orders - those that don't match restored entry_plans.
        Preserves legitimate market orders that have matching open_order_id in entry_plans.
        Returns count canceled.
        """
        open_orders = self.get_open_orders()
        market_buy_orders = [
            order for order in open_orders
            if order.get("type") == "market" and order.get("side") == "buy" and order.get("time_in_force") == "day"
        ]
        
        if not market_buy_orders:
            return 0
        
        # Build set of legitimate order IDs from restored entry_plans
        legitimate_order_ids = {
            plan.open_order_id 
            for plan in self.entry_plans.values() 
            if plan.open_order_id is not None
        }
        
        logger.info(f"Found {len(market_buy_orders)} open market buy orders, {len(legitimate_order_ids)} match restored entry_plans")
        
        canceled_count = 0
        preserved_count = 0
        
        for order in market_buy_orders:
            order_id = order.get("id")
            symbol = order.get("symbol")
            qty = order.get("qty")
            
            if order_id in legitimate_order_ids:
                # This order matches a restored entry_plan - preserve it
                logger.info(f"Preserving legitimate market order: {symbol} {qty} shares (ID: {order_id})")
                preserved_count += 1
            else:
                # This order is orphaned - cancel it
                if self._cancel_order(order_id):
                    logger.warning(f"Canceled orphaned market order: {symbol} {qty} shares (ID: {order_id})")
                    canceled_count += 1
                else:
                    logger.error(f"Failed to cancel orphaned market order: {symbol} {qty} shares (ID: {order_id})")
        
        if preserved_count > 0:
            logger.info(f"Preserved {preserved_count} legitimate market orders matching restored entry_plans")
        
        return canceled_count

    def get_broker_positions(self) -> Optional[List[dict]]:
        """Read live positions from Alpaca, independent of local bot memory.
        
        Returns:
            List of position dicts on success (may be empty if genuinely flat).
            None on API error — callers MUST check for None to avoid
            treating an API glitch as "broker is flat".
        """
        url = f"{self.base_url}/v2/positions"
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting broker positions: {e}")
            return None

    def broker_position_count(self) -> int:
        """Count live broker positions. Returns -1 if broker API failed."""
        positions = self.get_broker_positions()
        if positions is None:
            return -1
        return len(positions)

    def force_flatten_broker_positions(self, reason: str = "failsafe") -> Dict[str, object]:
        """Flatten ALL live broker positions using multi-layer retry.
        
        Retry chain per symbol:
        1. Market sell (full qty)
        2. Limit sell at -3% of last price (full qty)
        3. Limit sell at -5% of last price (half qty, then remaining)
        4. Flag for manual intervention
        
        Cancels all slicers and working orders first to prevent conflicts.
        """
        summary = {
            "reason": reason,
            "positions_seen": 0,
            "closes_submitted": 0,
            "fills_confirmed": 0,
            "symbols": [],
            "errors": [],
            "manual_required": [],
        }

        # Cancel all slicers first to prevent overlap (issue 4)
        if self.exit_slicers:
            logger.warning(f"Failsafe: canceling {len(self.exit_slicers)} active exit slicers")
            self.exit_slicers.clear()

        # Cancel any working orders
        self.cancel_all_open_orders()

        broker_positions = self.get_broker_positions()
        if broker_positions is None:
            logger.error(f"Broker API failed during {reason} flatten — cannot proceed, keeping local state")
            summary["errors"].append("broker API unreachable")
            return summary
        summary["positions_seen"] = len(broker_positions)

        if not broker_positions:
            logger.warning(f"No live broker positions found during {reason} flatten")
            if self.positions:
                logger.warning("Broker is flat but local positions remain; clearing local state")
                self.positions.clear()
            return summary

        logger.warning(f"Flattening {len(broker_positions)} live broker positions: {reason}")

        # Reset circuit breaker — this is last resort, try everything
        self._exit_failures.clear()

        for pos in broker_positions:
            try:
                symbol = pos.get("symbol")
                qty_raw = pos.get("qty", "0")
                qty = abs(int(float(qty_raw)))
                if not symbol or qty <= 0:
                    continue

                remaining_qty = qty
                total_filled = 0

                # ── Layer 1: Market sell ──
                sell_resp = self._submit_sell_order(symbol, remaining_qty)
                if sell_resp:
                    order_id = sell_resp.get("id")
                    if order_id:
                        summary["closes_submitted"] += 1
                        fill = self.get_order_fill(order_id, max_wait=30)
                        if fill:
                            filled = int(fill["filled_qty"])
                            total_filled += filled
                            remaining_qty -= filled
                            logger.warning(
                                f"FAILSAFE L1 {symbol}: market sell filled {filled}/{qty} "
                                f"@ {fill['filled_avg_price']:.4f}"
                            )

                # ── Layer 2: Limit sell at -3% ──
                if remaining_qty > 0:
                    last_price = self._get_last_price(symbol)
                    if last_price and last_price > 0:
                        limit_price = round(last_price * 0.97, 4)
                        logger.warning(f"FAILSAFE L2 {symbol}: trying limit sell {remaining_qty} @ {limit_price:.4f}")
                        sell_resp = self._submit_sell_order(symbol, remaining_qty, "limit", limit_price)
                        if sell_resp:
                            order_id = sell_resp.get("id")
                            if order_id:
                                summary["closes_submitted"] += 1
                                fill = self.get_order_fill(order_id, max_wait=20)
                                if fill:
                                    filled = int(fill["filled_qty"])
                                    total_filled += filled
                                    remaining_qty -= filled
                                    logger.warning(f"FAILSAFE L2 {symbol}: limit sell filled {filled}")

                # ── Layer 3: Limit sell at -5%, half qty then remainder ──
                if remaining_qty > 0:
                    last_price = self._get_last_price(symbol) or last_price
                    if last_price and last_price > 0:
                        limit_price = round(last_price * 0.95, 4)
                        half_qty = max(1, remaining_qty // 2)
                        logger.warning(f"FAILSAFE L3 {symbol}: trying limit sell {half_qty} @ {limit_price:.4f}")
                        sell_resp = self._submit_sell_order(symbol, half_qty, "limit", limit_price)
                        if sell_resp:
                            order_id = sell_resp.get("id")
                            if order_id:
                                summary["closes_submitted"] += 1
                                fill = self.get_order_fill(order_id, max_wait=15)
                                if fill:
                                    filled = int(fill["filled_qty"])
                                    total_filled += filled
                                    remaining_qty -= filled

                        # Try the rest if half worked
                        if remaining_qty > 0:
                            sell_resp = self._submit_sell_order(symbol, remaining_qty, "limit", limit_price)
                            if sell_resp:
                                order_id = sell_resp.get("id")
                                if order_id:
                                    summary["closes_submitted"] += 1
                                    fill = self.get_order_fill(order_id, max_wait=15)
                                    if fill:
                                        filled = int(fill["filled_qty"])
                                        total_filled += filled
                                        remaining_qty -= filled

                # ── Result tracking ──
                if total_filled > 0:
                    summary["fills_confirmed"] += 1
                    summary["symbols"].append(symbol)

                # Update local state
                if remaining_qty <= 0:
                    self.positions.pop(symbol, None)
                    self.exit_slicers.pop(symbol, None)
                    logger.info(f"FAILSAFE COMPLETE {symbol}: {total_filled}/{qty} shares closed")
                else:
                    # Reduce local qty to match what's still open
                    local_pos = self.positions.get(symbol)
                    if local_pos:
                        local_pos.quantity = max(0, local_pos.quantity - total_filled)

                    # ── Layer 4: Flag for manual intervention ──
                    logger.critical(
                        f"MANUAL INTERVENTION REQUIRED: {symbol} still has {remaining_qty} shares "
                        f"after all retry layers ({reason})"
                    )
                    summary["manual_required"].append(f"{symbol}: {remaining_qty} shares remaining")
                    summary["errors"].append(f"{symbol}: {remaining_qty}/{qty} unfilled after all layers")

            except Exception as e:
                msg = f"{pos.get('symbol', 'UNKNOWN')}: {e}"
                logger.error(f"Error flattening broker position - {msg}")
                summary["errors"].append(msg)

        return summary

    def nuclear_flatten(self, max_rounds: int = 5) -> Dict[str, object]:
        """NUCLEAR OPTION: spam market sells until broker is completely flat.
        
        Ignores circuit breakers entirely. Retries up to max_rounds.
        Each round: cancel all orders, re-read broker, market sell everything.
        
        Use only after 3:58 PM as absolute last resort.
        """
        logger.critical(f"NUCLEAR FLATTEN: starting — will retry up to {max_rounds} rounds")

        # Nuke all state that could interfere
        self.exit_slicers.clear()
        self._exit_failures.clear()
        self._exit_failure_times.clear()

        summary = {
            "rounds": 0,
            "total_sells": 0,
            "total_filled": 0,
            "still_open": [],
        }

        for rnd in range(1, max_rounds + 1):
            summary["rounds"] = rnd

            self.cancel_all_open_orders()
            time.sleep(1)

            broker_positions = self.get_broker_positions()
            if broker_positions is None:
                logger.critical(f"NUCLEAR FLATTEN round {rnd}: broker API failed — retrying")
                time.sleep(2)
                continue
            if not broker_positions:
                logger.critical(f"NUCLEAR FLATTEN: broker is FLAT after round {rnd}")
                self.positions.clear()
                return summary

            logger.critical(f"NUCLEAR FLATTEN round {rnd}: {len(broker_positions)} positions remaining")

            for pos in broker_positions:
                try:
                    symbol = pos.get("symbol")
                    qty = abs(int(float(pos.get("qty", "0"))))
                    if not symbol or qty <= 0:
                        continue

                    # Bypass _submit_sell_order to avoid circuit breaker entirely
                    url = f"{self.base_url}/v2/orders"
                    order_data = {
                        "symbol": symbol,
                        "qty": str(qty),
                        "side": "sell",
                        "type": "market",
                        "time_in_force": "day",
                    }
                    try:
                        resp = self.session.post(url, json=order_data, timeout=10)
                        resp.raise_for_status()
                        order_id = resp.json().get("id")
                        summary["total_sells"] += 1
                        logger.critical(f"NUCLEAR SELL {symbol} x{qty} (ID: {order_id})")

                        if order_id:
                            fill = self.get_order_fill(order_id, max_wait=15)
                            if fill and int(fill["filled_qty"]) > 0:
                                summary["total_filled"] += int(fill["filled_qty"])
                                self.positions.pop(symbol, None)
                    except Exception as e:
                        logger.critical(f"NUCLEAR SELL FAILED {symbol}: {e}")
                except Exception as e:
                    logger.critical(f"NUCLEAR FLATTEN error: {e}")

            time.sleep(2)

        # Final check
        remaining = self.get_broker_positions()
        if remaining is None:
            logger.critical(f"NUCLEAR FLATTEN: broker API failed on final check — cannot confirm flat")
        elif remaining:
            symbols = [p.get("symbol") for p in remaining]
            summary["still_open"] = symbols
            logger.critical(f"NUCLEAR FLATTEN INCOMPLETE: {symbols} still open after {max_rounds} rounds")
        else:
            self.positions.clear()
            logger.critical(f"NUCLEAR FLATTEN: broker confirmed flat after {max_rounds} rounds")

        return summary

    def reconcile_local_positions_from_broker(self) -> Dict[str, str]:
        """Sync local positions to match broker ground truth.
        
        For each broker position:
        - Missing locally → create from broker data.
        - Exists locally with wrong quantity → correct to broker quantity.
        - Exists locally with correct quantity → no change.
        
        Also removes local positions that the broker no longer holds.
        
        Returns dict of {symbol: action} for logging (added/corrected/removed).
        """
        broker_positions = self.get_broker_positions()
        actions = {}

        if broker_positions is None:
            logger.error("Broker API failed during reconciliation — skipping to preserve local state")
            return actions

        broker_map = {}
        if broker_positions:
            for pos in broker_positions:
                symbol = pos.get("symbol")
                qty = int(abs(float(pos.get("qty", 0))))
                avg_entry = float(pos.get("avg_entry_price", 0) or 0)
                if symbol and qty > 0:
                    broker_map[symbol] = {"qty": qty, "avg_entry": avg_entry}

        # Add missing + correct wrong quantities
        for symbol, bdata in broker_map.items():
            qty = bdata["qty"]
            avg_entry = bdata["avg_entry"]

            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    entry_price=avg_entry,
                    quantity=qty,
                    entry_time=datetime.now(),
                    entry_gap_pct=0.0,
                    adv_estimate=0.0,
                    peak_price=avg_entry,
                    current_price=avg_entry,
                )
                logger.warning(f"Broker reconcile: ADDED {symbol} qty={qty} avg={avg_entry:.4f}")
                actions[symbol] = "added"
            else:
                local_pos = self.positions[symbol]
                if local_pos.quantity != qty:
                    old_qty = local_pos.quantity
                    local_pos.quantity = qty
                    local_pos.entry_price = avg_entry
                    logger.warning(f"Broker reconcile: CORRECTED {symbol} qty {old_qty} → {qty}, avg → {avg_entry:.4f}")
                    actions[symbol] = "corrected"

        # Remove local positions the broker no longer holds
        local_only = [s for s in list(self.positions.keys()) if s not in broker_map]
        for symbol in local_only:
            del self.positions[symbol]
            self.exit_slicers.pop(symbol, None)
            logger.warning(f"Broker reconcile: REMOVED {symbol} (broker no longer holds)")
            actions[symbol] = "removed"

        if actions:
            logger.info(f"Broker reconcile complete: {actions}")
        return actions
