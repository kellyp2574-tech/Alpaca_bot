"""Position manager for gap momentum strategy with VIX-conditioned exits"""
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
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
        self.entry_plans: Dict[str, EntryExecutionPlan] = {}
        self.entry_stage1_done = False
        self.entry_stage2_done = False

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
        """Calculate position size with liquidity cap (0.3% of ADV) and absolute caps"""
        liquidity_cap = adv * config.LIQUIDITY_CAP_PCT
        position_dollars = min(target_dollars, liquidity_cap, config.MAX_POSITION_DOLLARS)
        quantity = int(position_dollars / current_price)
        quantity = min(quantity, config.MAX_POSITION_SHARES)
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
                
                # On partial fill, optionally cancel immediately and poll until terminal state
                if status == "partially_filled" and filled_qty > 0 and filled_avg_price:
                    if not allow_partial_cancel:
                        # For open market orders: don't cancel aggressively, just wait for terminal state
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
                    # For open market orders: don't cancel on timeout, just return what we have
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

    def enter_positions_moo(self, candidates: List, vix_level: float, capital_override: Optional[float] = None) -> Tuple[List[Position], float]:
        """
        Enter MOO (Market On Open) orders with two-pass process:
        1. Submit all orders before 9:28 cutoff
        2. Poll fills after 9:30 auction
        
        Returns: (positions_entered, total_capital_used)
        """
        equity = self.get_account_equity()
        if equity <= 0:
            logger.error("Cannot enter positions: no equity")
            return [], 0.0
        
        if not candidates:
            return [], 0.0
        
        url = f"{self.base_url}/v2/account"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            account_data = response.json()
            # Use buying_power instead of cash to support margin accounts
            buying_power = float(account_data.get("buying_power", equity))
        except requests.exceptions.RequestException:
            buying_power = equity
        
        # If capital_override specified, use it (for filler phase after core deployment)
        available_capital = capital_override if capital_override is not None else buying_power
        
        # Sort candidates by ADV (lowest first) so liquidity-constrained names get allocated first
        # This allows spare funds to carry over to higher ADV candidates
        sorted_candidates = sorted(candidates, key=lambda c: c.adv_estimate)
        planned_slots = min(len(sorted_candidates), config.MAX_POSITIONS)
        selected_candidates = sorted_candidates[:planned_slots]
        
        logger.info(f"MOO Portfolio plan: {planned_slots} positions, ${available_capital:,.2f} capital (buying_power), sorted by ADV (lowest first)")
        
        # PHASE 1: Submit all MOO orders before cutoff
        submitted = []  # List of (candidate, expected_qty, order_id, reserved_budget, expected_budget)
        submitted_symbols = set()  # Track symbols submitted in this batch to prevent duplicates
        remaining_capital = available_capital
        remaining_slots = planned_slots
        
        for i, candidate in enumerate(selected_candidates):
            symbol = candidate.symbol
            
            # CRITICAL FIX: Skip if we already have a position in this symbol
            if symbol in self.positions:
                logger.warning(f"Skipping {symbol}: already have position ({self.positions[symbol].quantity} shares)")
                remaining_slots -= 1
                continue
            
            # CRITICAL FIX: Skip if already submitted in this batch (duplicate candidate check)
            if symbol in submitted_symbols:
                logger.warning(f"Skipping duplicate candidate in same batch: {symbol}")
                remaining_slots -= 1
                continue
            
            expected_price = candidate.open_price
            
            # Dynamic budget: remaining capital / remaining slots
            # This allows spare funds from liquidity-capped candidates to carry over
            per_position_budget = remaining_capital / remaining_slots if remaining_slots > 0 else 0
            
            if per_position_budget <= 0:
                logger.warning(f"Skipping {symbol}: zero budget remaining")
                continue
            
            # Calculate position size (may be capped by liquidity)
            quantity = self.calculate_position_size(per_position_budget, candidate.adv_estimate, expected_price)
            if quantity <= 0:
                logger.warning(f"Skipping {symbol}: liquidity cap prevents allocation (ADV=${candidate.adv_estimate:,.0f})")
                remaining_slots -= 1  # Still consume a slot, but capital remains available
                continue
            
            # Track actual budget used with conservative reservation for slippage
            # Reserve 2% extra to account for potential price movement between expected and actual fill
            expected_budget = quantity * expected_price
            reserved_budget = expected_budget * 1.02  # 2% slippage buffer
            
            # Submit market order (legacy path)
            order_id = self._submit_market_order(symbol, quantity, "buy")
            if not order_id:
                remaining_slots -= 1
                continue
            
            submitted.append((candidate, quantity, order_id, reserved_budget, expected_budget))
            submitted_symbols.add(symbol)  # Track symbol as submitted
            remaining_capital -= reserved_budget
            remaining_slots -= 1
            logger.info(f"MOO submitted [{len(submitted)}/{planned_slots}]: {symbol} {quantity} shares @ ${expected_price:.2f} (expected: ${expected_budget:,.2f}, reserved: ${reserved_budget:,.2f}, remaining: ${remaining_capital:,.2f})")
        
        logger.info(f"MOO submission phase complete: {len(submitted)}/{planned_slots} orders submitted, ${remaining_capital:,.2f} unallocated")
        
        # PHASE 2: Poll fills for all submitted orders
        entered = []
        total_allocated = 0.0
        
        for candidate, expected_qty, order_id, reserved_budget, expected_budget in submitted:
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
        
        logger.info(f"MOO entry complete: {len(entered)}/{len(submitted)} filled, total=${total_allocated:,.2f}, capital_remaining=${remaining_capital:,.2f}")
        return entered, total_allocated

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
        except requests.exceptions.RequestException as e:
            logger.error(f"Marketable limit order error for {symbol}: {e}")
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
        """
        Build an aggressive buy limit from already-fetched snapshot data.
        Uses ask if available, otherwise last price, with 50 bps buffer.
        """
        ask = snapshot.get("ask")
        last_price = snapshot.get("last_price") or snapshot.get("close") or fallback_open

        ref_price = ask or last_price
        if not ref_price or ref_price <= 0:
            return None

        return ref_price * (1 + config.POST_OPEN_BUY_LIMIT_BUFFER)

    def build_entry_plans(self, candidates: List, capital_override: Optional[float] = None) -> Dict[str, EntryExecutionPlan]:
        """
        Build target position sizes once, then split into open order + post-open remainder.
        Does NOT submit orders yet.
        """
        if not candidates:
            return {}

        # Skip account API calls when capital_override is provided (saves 2 API calls)
        if capital_override is not None:
            available_capital = capital_override
        else:
            equity = self.get_account_equity()
            if equity <= 0:
                logger.error("Cannot build entry plans: no equity")
                return {}

            url = f"{self.base_url}/v2/account"
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                account_data = response.json()
                available_capital = float(account_data.get("buying_power", equity))
            except requests.exceptions.RequestException:
                available_capital = equity

        # Sort candidates by ADV (lowest first) so liquidity-constrained names get allocated first
        sorted_candidates = sorted(candidates, key=lambda c: c.adv_estimate)
        planned_slots = min(len(sorted_candidates), config.MAX_POSITIONS)
        selected_candidates = sorted_candidates[:planned_slots]

        remaining_capital = available_capital
        remaining_slots = planned_slots
        plans: Dict[str, EntryExecutionPlan] = {}

        for candidate in selected_candidates:
            symbol = candidate.symbol
            expected_price = candidate.open_price

            if symbol in plans or symbol in self.positions:
                continue

            per_position_budget = remaining_capital / remaining_slots if remaining_slots > 0 else 0
            if per_position_budget <= 0:
                break

            target_qty = self.calculate_position_size(
                per_position_budget,
                candidate.adv_estimate,
                expected_price
            )
            if target_qty <= 0:
                remaining_slots -= 1
                continue

            # Calculate open order size - when OPEN_ENTRY_PCT=1.0, send full target size at open
            open_qty = int(target_qty * config.OPEN_ENTRY_PCT)
            if open_qty <= 0:
                open_qty = 0
            open_qty = min(open_qty, target_qty)
            planned_remaining_qty = target_qty - open_qty
            
            # When OPEN_ENTRY_PCT=1.0 (full size at open), no rescue needed
            if config.OPEN_ENTRY_PCT >= 1.0:
                planned_remaining_qty = 0

            expected_budget = target_qty * expected_price
            reserved_budget = expected_budget * 1.02

            plans[symbol] = EntryExecutionPlan(
                symbol=symbol,
                target_qty=target_qty,
                open_qty=open_qty,
                planned_remaining_qty=planned_remaining_qty,
                expected_open_price=expected_price,
                gap_pct=candidate.gap_pct,
                adv_estimate=candidate.adv_estimate,
            )

            remaining_capital -= reserved_budget
            remaining_slots -= 1

        self.entry_plans.update(plans)
        logger.info(f"Built {len(plans)} staged entry plans")
        return plans

    def submit_open_entry_orders(self, plans: Dict[str, EntryExecutionPlan], state_saver=None) -> None:
        """Submit full market orders at 9:30:00 for entry.
        
        When OPEN_ENTRY_PCT=1.0, submits full target_qty for pure market-at-open execution.
        
        Args:
            state_saver: Optional callable invoked after each successful submission
                         so order_ids are persisted incrementally (crash safety).
        """
        for symbol, plan in plans.items():
            # When OPEN_ENTRY_PCT=1.0, use target_qty directly; otherwise use open_qty
            submit_qty = plan.target_qty if config.OPEN_ENTRY_PCT >= 1.0 else plan.open_qty
            if submit_qty <= 0:
                continue
            if symbol in self.positions:
                continue

            order_id = self._submit_market_order(symbol, submit_qty, "buy")
            if order_id:
                plan.open_order_id = order_id
                plan.open_qty = submit_qty  # Update to actual submitted qty
                logger.info(f"Market order submitted at open: {symbol} {submit_qty}/{plan.target_qty}")
                if state_saver:
                    state_saver()

    def reconcile_open_order_fills(self) -> None:
        """Read market order outcomes via batch polling (all orders per cycle, not sequential)."""
        # Collect orders to poll
        active_orders = {}
        for symbol, plan in self.entry_plans.items():
            if not plan.open_order_id or plan.open_filled_qty > 0:
                continue
            active_orders[plan.open_order_id] = {"symbol": symbol, "plan": plan}

        if not active_orders:
            return

        logger.info(f"Reconciling {len(active_orders)} market orders (batch polling)...")
        terminal_states = {"filled", "canceled", "done_for_day", "expired", "rejected"}
        start_time = datetime.now()
        max_wait = 10  # seconds total, not per-order

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

                    if plan.open_filled_avg_price > 0:
                        plan.expected_open_price = plan.open_filled_avg_price
                        logger.info(f"Market order reconciled: {symbol} filled {filled_qty}/{plan.open_qty}, updated ref price to {plan.expected_open_price:.4f}")
                    elif filled_qty > 0:
                        logger.info(f"Market order reconciled: {symbol} filled {filled_qty}/{plan.open_qty}")
                    else:
                        logger.info(f"Market order reconciled: {symbol} no fill (status={status})")
                    completed.append(order_id)

            for order_id in completed:
                active_orders.pop(order_id, None)

            if active_orders:
                time.sleep(1.0)

        # Any still-active orders after timeout: mark as no-fill
        for order_id, meta in active_orders.items():
            symbol = meta["symbol"]
            logger.info(f"Market order reconciled: {symbol} no fill (poll timeout)")

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
        """
        Submit aggressive post-open marketable limits for remaining entry size.
        pass_name: 'market1' or 'market2'
        
        Args:
            state_saver: Optional callable invoked after each successful submission
                         so rescue order_ids are persisted incrementally (crash safety).
        
        FIXES APPLIED:
        - Parallel submission: submit all orders first, then poll fills
        - Buying power check: scale quantities based on actual available capital
        - Duplicate protection: skip if order_id already exists for this pass
        - Batched snapshots: fetch all at once instead of per-symbol
        """
        # STEP 1: Check current buying power and calculate scale factor
        account_capital = self.get_total_capital()
        
        # Calculate total planned remaining capital across all plans
        total_planned_remaining = 0.0
        for plan in self.entry_plans.values():
            if plan.finalized or plan.symbol in self.positions:
                continue
            already_filled = plan.open_filled_qty + plan.market1_filled_qty + plan.market2_filled_qty
            remaining_qty = plan.target_qty - already_filled
            if remaining_qty > 0:
                # Use expected open price as conservative estimate
                total_planned_remaining += remaining_qty * plan.expected_open_price
        
        # Calculate scale factor (never exceed 1.0, reduce if needed)
        scale_factor = 1.0
        if total_planned_remaining > 0 and account_capital > 0:
            scale_factor = min(1.0, account_capital / total_planned_remaining)
            if scale_factor < 1.0:
                logger.warning(f"Rescue pass scaling down to {scale_factor:.1%} due to buying power constraints")
        
        # STEP 2: Collect all symbols to process and fetch batched snapshots
        symbols_to_process = []
        for symbol, plan in self.entry_plans.items():
            if plan.finalized or plan.symbol in self.positions:
                continue
            # CRITICAL FIX: Skip if already submitted for this pass (duplicate protection)
            if pass_name == "market1" and plan.market1_order_id:
                continue
            if pass_name == "market2" and plan.market2_order_id:
                continue
            symbols_to_process.append(symbol)
        
        if not symbols_to_process:
            logger.info(f"No symbols to process for {pass_name} rescue pass")
            return
        
        # Batch fetch snapshots for all symbols at once
        all_snapshots = self.client.get_snapshots(symbols_to_process)
        
        # STEP 3: Submit all orders first (no blocking)
        submitted_orders = []  # List of (symbol, order_id, remaining_qty, plan)
        
        for symbol in symbols_to_process:
            plan = self.entry_plans[symbol]
            snapshot = all_snapshots.get(symbol, {}) if all_snapshots else {}
            
            already_filled = plan.open_filled_qty + plan.market1_filled_qty + plan.market2_filled_qty
            remaining_qty = plan.target_qty - already_filled
            
            if remaining_qty < config.MIN_RESCUE_SHARES:
                continue
            
            # Apply scale factor based on available buying power
            if scale_factor < 1.0:
                scaled_qty = int(remaining_qty * scale_factor)
                if scaled_qty < config.MIN_RESCUE_SHARES:
                    continue  # Skip if scaled quantity too small
                remaining_qty = scaled_qty
            
            last_price = snapshot.get("last_price") or snapshot.get("close") or plan.expected_open_price
            if not last_price or last_price <= 0:
                continue
            
            # Chase guard - validate expected_open_price to prevent divide-by-zero
            if not plan.expected_open_price or plan.expected_open_price <= 0:
                logger.warning(f"Skipping rescue for {symbol}: invalid expected_open_price={plan.expected_open_price}")
                continue
            
            chase_pct = (last_price - plan.expected_open_price) / plan.expected_open_price
            if chase_pct > config.MAX_CHASE_FROM_OPEN_PCT:
                logger.warning(f"Skipping rescue for {symbol}: chase {chase_pct:.2%} > max")
                continue
            
            if remaining_qty * last_price < config.MIN_RESCUE_NOTIONAL:
                continue
            
            # FIX: Use snapshot-based helper instead of fetching fresh per-symbol
            limit_price = self._get_aggressive_buy_limit_from_snapshot(snapshot, plan.expected_open_price)
            if not limit_price:
                continue
            
            order_id = self._submit_marketable_limit_order(symbol, remaining_qty, "buy", limit_price)
            if order_id:
                # FIX: Store order ID immediately for recovery safety
                if pass_name == "market1":
                    plan.market1_order_id = order_id
                else:
                    plan.market2_order_id = order_id
                
                submitted_orders.append((symbol, order_id, remaining_qty, limit_price, plan))
                logger.info(f"{pass_name} order submitted: {symbol} {remaining_qty} shares @ {limit_price:.4f}")
                if state_saver:
                    state_saver()
        
        if not submitted_orders:
            logger.info(f"No orders submitted for {pass_name} rescue pass")
            return
        
        # STEP 4: True parallel-style polling for all submitted orders
        logger.info(f"{pass_name}: Polling fills for {len(submitted_orders)} orders...")
        
        # Setup parallel polling structure
        start_time = datetime.now()
        max_wait = 10
        
        active_orders = {
            order_id: {
                "symbol": symbol,
                "requested_qty": requested_qty,
                "limit_price": limit_price,
                "plan": plan,
                "cancel_requested": False,
            }
            for symbol, order_id, requested_qty, limit_price, plan in submitted_orders
        }
        
        results = {}
        terminal_states = {"filled", "canceled", "done_for_day", "expired", "rejected"}
        
        while active_orders and (datetime.now() - start_time).total_seconds() < max_wait:
            completed = []
            
            for order_id, meta in list(active_orders.items()):
                url = f"{self.base_url}/v2/orders/{order_id}"
                
                try:
                    response = self.session.get(url, timeout=5)
                    response.raise_for_status()
                    order = response.json()
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Polling error {order_id}: {e}")
                    continue
                
                status = order.get("status")
                filled_qty = float(order.get("filled_qty", 0))
                filled_avg_price = order.get("filled_avg_price")
                
                if status in terminal_states:
                    results[order_id] = {
                        "filled_qty": int(filled_qty),
                        "filled_avg_price": float(filled_avg_price) if filled_avg_price else 0.0,
                        "status": status,
                    }
                    completed.append(order_id)
                elif status == "partially_filled" and filled_qty > 0:
                    # Same behavior as existing logic - cancel on partial, but only once
                    if not meta["cancel_requested"]:
                        self._cancel_order(order_id)
                        meta["cancel_requested"] = True
            
            for order_id in completed:
                active_orders.pop(order_id, None)
            
            time.sleep(1.0)
        
        # Final cleanup (timeout case) - use get_order_fill for safer handling
        for order_id, meta in active_orders.items():
            fill = self.get_order_fill(order_id, max_wait=5, allow_partial_cancel=True)
            if fill:
                results[order_id] = {
                    "filled_qty": int(fill.get("filled_qty", 0)),
                    "filled_avg_price": float(fill.get("filled_avg_price", 0.0)),
                    "status": fill.get("status", "timeout"),
                }
            else:
                results[order_id] = {
                    "filled_qty": 0,
                    "filled_avg_price": 0.0,
                    "status": "timeout",
                }
        
        # Map results back to plans
        for symbol, order_id, requested_qty, limit_price, plan in submitted_orders:
            fill = results.get(order_id, {})
            filled_qty = int(fill.get("filled_qty", 0))
            filled_avg_price = float(fill.get("filled_avg_price", 0.0))
            
            # Only store fill quantities (order_id already stored above)
            if pass_name == "market1":
                plan.market1_filled_qty = filled_qty
                plan.market1_filled_avg_price = filled_avg_price
            else:
                plan.market2_filled_qty = filled_qty
                plan.market2_filled_avg_price = filled_avg_price
            
            avg_fill_str = f"{filled_avg_price:.4f}" if filled_avg_price > 0 else "N/A"
            logger.info(
                f"{pass_name} rescue complete: {symbol} requested {requested_qty}, filled {filled_qty}, "
                f"limit={limit_price:.4f}, avg_fill={avg_fill_str}"
            )

    def finalize_entry_positions(self) -> List[Position]:
        """Create final Position objects from all entry fills."""
        entered = []

        for symbol, plan in self.entry_plans.items():
            if plan.finalized:
                continue

            total_qty = plan.open_filled_qty + plan.market1_filled_qty + plan.market2_filled_qty
            if total_qty <= 0:
                plan.finalized = True
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
            if entry_price <= 0:
                logger.warning(f"Skipping trailing stop calc for {symbol}: entry_price={entry_price}")
                continue

            gain_pct = (current_price - entry_price) / entry_price

            if gain_pct >= config.TRAILING_STOP_ACTIVATION:
                position.is_trailing_active = True

            if position.is_trailing_active:
                trail_level = position.peak_price * (1 - config.TRAILING_STOP_PCT)
                position.trailing_stop_price = max(position.trailing_stop_price, trail_level)

        return current_prices

    def check_exits(self, current_time: dt_time, vix_level: float, current_prices: Dict[str, float]) -> List[str]:
        """Check if positions should be exited based on trailing stops (full) or time (sliced)"""
        exited = []

        # Determine exit window based on VIX regime
        if vix_level < config.VIX_LOW_THRESHOLD:
            target_exit = datetime.strptime(config.EXIT_TIME_LOW_VIX, "%H:%M").time()
        elif vix_level > config.VIX_HIGH_THRESHOLD:
            target_exit = datetime.strptime(config.EXIT_TIME_HIGH_VIX, "%H:%M").time()
        else:
            target_exit = datetime.strptime(config.EXIT_TIME_MIDDLE_VIX, "%H:%M").time()

        target_dt = datetime.combine(datetime.now().date(), target_exit)
        exit_window_start = (target_dt - timedelta(minutes=1)).time()
        exit_window_end = (target_dt + timedelta(minutes=1)).time()

        current_dt = datetime.combine(datetime.now().date(), current_time)

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

    def _start_exit_slicer(self, symbol: str, current_time: dt_time, vix_level: float, reason: str):
        """Start a time-based exit slicer for gradual position liquidation"""
        if symbol in self.exit_slicers:
            return

        # Determine slice parameters based on VIX regime
        if vix_level > config.VIX_HIGH_THRESHOLD:
            total_window_seconds = 6 * 60   # 3:30 regime: 6 minutes
            slices = 3
        elif vix_level < config.VIX_LOW_THRESHOLD:
            total_window_seconds = 10 * 60  # 2:30 regime: 10 minutes
            slices = 3
        else:
            total_window_seconds = 8 * 60
            slices = 3

        now_dt = datetime.combine(datetime.now().date(), current_time)
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

    def get_broker_positions(self) -> List[dict]:
        """Read live positions from Alpaca, independent of local bot memory."""
        url = f"{self.base_url}/v2/positions"
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting broker positions: {e}")
            return []

    def broker_position_count(self) -> int:
        """Count live broker positions."""
        return len(self.get_broker_positions())

    def force_flatten_broker_positions(self, reason: str = "failsafe") -> Dict[str, object]:
        """
        Flatten ALL live broker positions based on Alpaca account state,
        not local self.positions.
        """
        summary = {
            "reason": reason,
            "positions_seen": 0,
            "orders_submitted": 0,
            "symbols": [],
            "errors": [],
        }

        # Best effort: cancel any working orders first
        self.cancel_all_open_orders()

        broker_positions = self.get_broker_positions()
        summary["positions_seen"] = len(broker_positions)

        if not broker_positions:
            logger.warning(f"No live broker positions found during {reason} flatten")
            # Also clear stale local positions if broker is flat
            if self.positions:
                logger.warning("Broker is flat but local positions remain; clearing local state")
                self.positions.clear()
            self.exit_slicers.clear()
            return summary

        logger.warning(f"Flattening {len(broker_positions)} live broker positions: {reason}")

        for pos in broker_positions:
            try:
                symbol = pos.get("symbol")
                qty_raw = pos.get("qty", "0")
                side = (pos.get("side") or "long").lower()

                qty = abs(int(float(qty_raw)))
                if not symbol or qty <= 0:
                    continue

                order_side = "sell" if side == "long" else "buy"

                order_id = self._submit_market_order(symbol, qty, order_side)
                if not order_id:
                    msg = f"{symbol}: failed to submit {order_side} {qty}"
                    logger.error(msg)
                    summary["errors"].append(msg)
                    continue

                fill = self.get_order_fill(order_id, max_wait=30)
                if fill:
                    filled_qty = int(fill["filled_qty"])
                    status = fill.get("status", "unknown")

                    logger.warning(
                        f"FAILSAFE FLATTEN {symbol}: {order_side} {filled_qty} "
                        f"@ {fill['filled_avg_price']:.4f} status={status}"
                    )

                    # Clear local state if filled_qty >= qty (broker exposure is gone)
                    # This handles all statuses: filled, timeout_with_fill, canceled_with_fill, etc.
                    if filled_qty >= qty:
                        self.positions.pop(symbol, None)
                        self.exit_slicers.pop(symbol, None)
                        logger.info(f"Cleared local state for {symbol}: {filled_qty}/{qty} shares filled")
                    else:
                        msg = f"{symbol}: partial fill {filled_qty}/{qty} shares, status={status}"
                        logger.warning(msg)
                        summary["errors"].append(msg)
                else:
                    msg = f"{symbol}: no fill confirmation for order {order_id}"
                    logger.error(msg)
                    summary["errors"].append(msg)

                summary["orders_submitted"] += 1
                summary["symbols"].append(symbol)

            except Exception as e:
                msg = f"{pos.get('symbol', 'UNKNOWN')}: {e}"
                logger.error(f"Error flattening broker position - {msg}")
                summary["errors"].append(msg)

        return summary

    def reconcile_local_positions_from_broker(self):
        """
        Rebuild missing local Position objects from broker positions
        when local state is empty or incomplete. Call this on startup
        after detecting broker positions that aren't in local memory.
        """
        broker_positions = self.get_broker_positions()
        if not broker_positions:
            return

        for pos in broker_positions:
            symbol = pos.get("symbol")
            qty = int(abs(float(pos.get("qty", 0))))
            avg_entry = float(pos.get("avg_entry_price", 0) or 0)

            if not symbol or qty <= 0:
                continue

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
                logger.warning(f"Rebuilt local position from broker: {symbol} qty={qty} avg={avg_entry:.4f}")
