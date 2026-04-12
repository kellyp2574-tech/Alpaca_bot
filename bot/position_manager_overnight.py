"""Position manager — overnight momentum strategy only.

This file contains ONLY the methods actively used by the overnight momentum bot.
Legacy gap-strategy / VIX-exit code lives in position_manager_legacy.py.
"""
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
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


class PositionManager:
    """Manages positions for overnight momentum strategy"""

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

        # Circuit breaker: track consecutive sell failures per symbol
        self._exit_failures: Dict[str, int] = {}
        self._exit_failure_times: Dict[str, datetime] = {}
        self._max_exit_failures = 3
        self._exit_cooldown_seconds = 60

    # ────────────────────────────────────────────────────────
    # Position loading / account queries
    # ────────────────────────────────────────────────────────

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

    def get_position_count(self) -> int:
        return len(self.positions)

    # ────────────────────────────────────────────────────────
    # Position sizing
    # ────────────────────────────────────────────────────────

    def calculate_position_size(self, target_dollars: float, adv: float, current_price: float) -> int:
        """Calculate position size with ADV cap and absolute dollar cap."""
        adv_dollar_cap = adv * config.ADV_CAP_PCT
        position_dollars = min(target_dollars, adv_dollar_cap, config.MAX_POSITION_DOLLARS)
        quantity = int(position_dollars / current_price) if current_price > 0 else 0
        return max(0, quantity)

    # ────────────────────────────────────────────────────────
    # Order submission
    # ────────────────────────────────────────────────────────

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

    def _submit_sell_order(self, symbol: str, qty: int, order_type: str = "market",
                           limit_price: Optional[float] = None) -> Optional[dict]:
        """Submit a sell order via POST /v2/orders.

        ALL exits go through this method — never DELETE /v2/positions.
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

    def _get_last_price(self, symbol: str) -> Optional[float]:
        """Get last trade price for a symbol (for limit sell fallback)."""
        try:
            snapshots = self.client.get_snapshots([symbol])
            snap = snapshots.get(symbol, {}) if snapshots else {}
            return snap.get("last_price") or snap.get("close")
        except Exception:
            return None

    # ────────────────────────────────────────────────────────
    # Order management
    # ────────────────────────────────────────────────────────

    def _cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully canceled or already complete."""
        url = f"{self.base_url}/v2/orders/{order_id}"
        try:
            response = self.session.delete(url, timeout=10)
            if response.status_code in (200, 204):
                logger.info(f"Canceled order {order_id}")
                return True
            elif response.status_code == 422:
                logger.info(f"Order {order_id} already complete (422)")
                return True
            else:
                logger.warning(f"Failed to cancel order {order_id}: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    def get_order_fill(self, order_id: str, max_wait: int = 30, allow_partial_cancel: bool = True) -> Optional[dict]:
        """Poll order until filled, partially filled, or timeout.

        On partial fill, cancel residual immediately inside this function,
        then re-read order state to return final filled quantity.
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

                if status == "partially_filled" and filled_qty > 0 and filled_avg_price:
                    if not allow_partial_cancel:
                        logger.info(f"Order {order_id} partially filled: {filled_qty} shares - waiting for terminal state (allow_partial_cancel=False)")
                        time.sleep(3.0)
                        continue

                    logger.info(f"Order {order_id} partially filled: {filled_qty} shares - waiting 10s for more fills")
                    time.sleep(10)

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
                    self._cancel_order(order_id)

                    terminal_states = {"filled", "canceled", "done_for_day", "expired", "rejected"}
                    poll_start = datetime.now()
                    while (datetime.now() - poll_start).total_seconds() < 10:
                        time.sleep(1.5)
                        try:
                            response = self.session.get(url, timeout=10)
                            response.raise_for_status()
                            final_order = response.json()
                            final_status = final_order.get("status", "unknown")

                            if final_status in terminal_states:
                                final_qty = float(final_order.get("filled_qty", filled_qty))
                                final_price = final_order.get("filled_avg_price", filled_avg_price)
                                return {
                                    "order_id": order_id,
                                    "filled_qty": final_qty,
                                    "filled_avg_price": float(final_price),
                                    "status": "filled" if final_status == "filled" else "partially_filled"
                                }
                        except Exception as e:
                            logger.warning(f"Error polling after cancellation: {e}")
                            break

                    return {"order_id": order_id, "filled_qty": filled_qty, "filled_avg_price": float(filled_avg_price), "status": "partially_filled"}

                if status in ("canceled", "expired", "rejected"):
                    if filled_qty > 0 and filled_avg_price:
                        logger.warning(f"Order {order_id} {status} but {filled_qty} shares filled")
                        return {"order_id": order_id, "filled_qty": filled_qty, "filled_avg_price": float(filled_avg_price), "status": f"{status}_with_fill"}
                    logger.error(f"Order {order_id} failed with status: {status}")
                    return None

                time.sleep(3.0)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error polling order {order_id}: {e}")
                return None

        # Timeout — check for partial fills
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            order = response.json()
            filled_qty = float(order.get("filled_qty", 0))
            filled_avg_price = order.get("filled_avg_price")
            if filled_qty > 0 and filled_avg_price:
                if not allow_partial_cancel:
                    return {"order_id": order_id, "filled_qty": filled_qty, "filled_avg_price": float(filled_avg_price), "status": "timeout_with_fill"}

                logger.info(f"Order {order_id} timeout with {filled_qty} shares filled - waiting 10s")
                time.sleep(10)

                try:
                    recheck = self.session.get(url, timeout=10).json()
                    recheck_status = recheck.get("status")
                    recheck_qty = float(recheck.get("filled_qty", 0))
                    if recheck_status == "filled" and recheck_qty > 0:
                        return {"order_id": order_id, "filled_qty": recheck_qty, "filled_avg_price": float(recheck.get("filled_avg_price")), "status": "filled"}
                    if recheck_qty > filled_qty:
                        filled_qty = recheck_qty
                        filled_avg_price = recheck.get("filled_avg_price", filled_avg_price)
                except Exception:
                    pass

                self._cancel_order(order_id)

                terminal_states = {"filled", "canceled", "done_for_day", "expired", "rejected"}
                poll_start = datetime.now()
                while (datetime.now() - poll_start).total_seconds() < 10:
                    time.sleep(1.5)
                    try:
                        response = self.session.get(url, timeout=10)
                        response.raise_for_status()
                        final_order = response.json()
                        final_status = final_order.get("status", "unknown")
                        if final_status in terminal_states:
                            final_qty = float(final_order.get("filled_qty", filled_qty))
                            final_price = final_order.get("filled_avg_price", filled_avg_price)
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

    # ────────────────────────────────────────────────────────
    # Exit logic
    # ────────────────────────────────────────────────────────

    def _is_exit_blocked(self, symbol: str) -> bool:
        """Check if a symbol is in cooldown after hitting the circuit breaker."""
        failures = self._exit_failures.get(symbol, 0)
        if failures < self._max_exit_failures:
            return False
        last_fail = self._exit_failure_times.get(symbol)
        if last_fail and (datetime.now() - last_fail).total_seconds() >= self._exit_cooldown_seconds:
            logger.info(f"Circuit breaker cooldown elapsed for {symbol} — allowing retry")
            self._exit_failures[symbol] = 0
            self._exit_failure_times.pop(symbol, None)
            return False
        remaining = self._exit_cooldown_seconds - (datetime.now() - last_fail).total_seconds() if last_fail else 0
        logger.debug(f"Circuit breaker active for {symbol}: {failures} failures, {remaining:.0f}s cooldown remaining")
        return True

    def _exit_position(self, symbol: str, reason: str):
        """Exit a full position via market sell with limit fallback + partial resubmit."""
        position = self.positions.get(symbol)
        if not position:
            return

        if self._is_exit_blocked(symbol):
            return

        qty = position.quantity
        sell_resp = self._submit_sell_order(symbol, qty)

        if not sell_resp:
            last_price = self._get_last_price(symbol)
            if last_price and last_price > 0:
                limit_price = round(last_price * 0.97, 4)
                logger.warning(f"Market sell failed for {symbol}, trying limit sell @ {limit_price:.4f}")
                sell_resp = self._submit_sell_order(symbol, qty, "limit", limit_price)

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
            # Resubmit for residual
            slice_residual = qty - filled_qty
            if slice_residual > 0 and fill_status != "filled":
                logger.warning(f"EXIT {symbol}: partial fill {filled_qty}/{qty} — resubmitting {slice_residual}")
                resub_resp = self._submit_sell_order(symbol, slice_residual)
                if resub_resp:
                    resub_id = resub_resp.get("id")
                    if resub_id:
                        resub_fill = self.get_order_fill(resub_id, max_wait=15)
                        if resub_fill:
                            resub_qty = int(resub_fill["filled_qty"])
                            remaining -= resub_qty
                            position.quantity = remaining

            if remaining <= 0:
                logger.info(f"FINAL EXIT {symbol} (resubmit completed): {reason}, status={fill_status}")
                self.positions.pop(symbol, None)
            else:
                logger.info(
                    f"EXIT {symbol}: {filled_qty} @ {exit_price:.2f} "
                    f"(P&L: {pnl:+.2f}, {pnl_pct:+.1f}%) - remaining {remaining}, status={fill_status}"
                )
        else:
            logger.info(
                f"FINAL EXIT {symbol}: {filled_qty} @ {exit_price:.2f} "
                f"(P&L: {pnl:+.2f}, {pnl_pct:+.1f}%) - {reason}, status={fill_status}"
            )
            del self.positions[symbol]

    def force_exit_all(self, reason: str = "force"):
        logger.warning(f"Force exiting all positions: {reason}")
        for symbol in list(self.positions.keys()):
            self._exit_position(symbol, reason)

    # ────────────────────────────────────────────────────────
    # Order cancellation
    # ────────────────────────────────────────────────────────

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

    # ────────────────────────────────────────────────────────
    # Broker interaction
    # ────────────────────────────────────────────────────────

    def get_broker_positions(self) -> Optional[List[dict]]:
        """Read live positions from Alpaca, independent of local bot memory.

        Returns:
            List of position dicts on success (may be empty if genuinely flat).
            None on API error.
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

    def reconcile_local_positions_from_broker(self) -> Dict[str, str]:
        """Sync local positions to match broker ground truth."""
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

        local_only = [s for s in list(self.positions.keys()) if s not in broker_map]
        for symbol in local_only:
            del self.positions[symbol]
            logger.warning(f"Broker reconcile: REMOVED {symbol} (broker no longer holds)")
            actions[symbol] = "removed"

        if actions:
            logger.info(f"Broker reconcile complete: {actions}")
        return actions

    # ────────────────────────────────────────────────────────
    # Failsafe flatten
    # ────────────────────────────────────────────────────────

    def force_flatten_broker_positions(self, reason: str = "failsafe") -> Dict[str, object]:
        """Flatten ALL live broker positions using multi-layer retry.

        Retry chain per symbol:
        1. Market sell (full qty)
        2. Limit sell at -3% of last price (full qty)
        3. Limit sell at -5% of last price (half qty, then remaining)
        4. Flag for manual intervention
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

        self.cancel_all_open_orders()

        broker_positions = self.get_broker_positions()
        if broker_positions is None:
            logger.error(f"Broker API failed during {reason} flatten — cannot proceed")
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

                # Layer 1: Market sell
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

                # Layer 2: Limit sell at -3%
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

                # Layer 3: Limit sell at -5%, half then rest
                if remaining_qty > 0:
                    last_price = self._get_last_price(symbol) or last_price
                    if last_price and last_price > 0:
                        limit_price = round(last_price * 0.95, 4)
                        half_qty = max(1, remaining_qty // 2)
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

                if total_filled > 0:
                    summary["fills_confirmed"] += 1
                    summary["symbols"].append(symbol)

                if remaining_qty <= 0:
                    self.positions.pop(symbol, None)
                    logger.info(f"FAILSAFE COMPLETE {symbol}: {total_filled}/{qty} shares closed")
                else:
                    local_pos = self.positions.get(symbol)
                    if local_pos:
                        local_pos.quantity = max(0, local_pos.quantity - total_filled)
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
        Use only after 3:58 PM as absolute last resort.
        """
        logger.critical(f"NUCLEAR FLATTEN: starting — will retry up to {max_rounds} rounds")

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

        remaining = self.get_broker_positions()
        if remaining is None:
            logger.critical(f"NUCLEAR FLATTEN: broker API failed on final check")
        elif remaining:
            symbols = [p.get("symbol") for p in remaining]
            summary["still_open"] = symbols
            logger.critical(f"NUCLEAR FLATTEN INCOMPLETE: {symbols} still open after {max_rounds} rounds")
        else:
            self.positions.clear()
            logger.critical(f"NUCLEAR FLATTEN: broker confirmed flat after {max_rounds} rounds")

        return summary
