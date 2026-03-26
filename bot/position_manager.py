"""Position manager for gap momentum strategy with VIX-conditioned exits"""
import logging
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
    current_price: float = field(default=0.0)  # Last known price from update_positions


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

        # Track which positions have been scheduled for timed exit (distributed over 3 minutes)
        self.timed_exit_scheduled: Dict[str, datetime] = {}

    def load_positions(self, saved_positions: Dict):
        """Restore positions from saved state including trailing stop data"""
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
                logger.info(f"Restored position: {symbol} {position.quantity} shares @ {position.entry_price:.2f} (trailing_active={position.is_trailing_active})")
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

    def calculate_position_size(
        self, 
        target_dollars: float,  # Target dollar allocation for this position
        adv: float,  # Average daily volume
        current_price: float  # Entry price
    ) -> int:
        """
        Calculate position size based on:
        1. Target dollar allocation (from portfolio construction)
        2. Liquidity cap: max 0.3% of ADV
        """
        # Liquidity cap (0.3% of ADV)
        liquidity_cap = adv * config.LIQUIDITY_CAP_PCT
        
        # Use the smaller of target and liquidity cap
        position_dollars = min(target_dollars, liquidity_cap)
        
        # Convert to shares
        quantity = int(position_dollars / current_price)
        
        return max(0, quantity)

    def get_order_fill(self, order_id: str, max_wait: int = 30) -> Optional[dict]:
        """Poll order until filled or timeout. Returns fill data with actual price."""
        url = f"{self.base_url}/v2/orders/{order_id}"
        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < max_wait:
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                order = response.json()

                status = order.get("status")
                filled_qty = float(order.get("filled_qty", 0))
                filled_avg_price = order.get("filled_avg_price")

                if status == "filled" and filled_qty > 0 and filled_avg_price:
                    return {
                        "order_id": order_id,
                        "filled_qty": filled_qty,
                        "filled_avg_price": float(filled_avg_price),
                        "status": "filled",
                    }
                elif status in ("canceled", "expired", "rejected"):
                    logger.error(f"Order {order_id} failed with status: {status}")
                    return None

                # Wait before polling again
                import time
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error polling order {order_id}: {e}")
                return None

        logger.warning(f"Order {order_id} did not fill within {max_wait} seconds")
        return None

    def enter_positions(
        self, candidates: List, vix_level: float
    ) -> List[Position]:
        """
        Enter market orders at 9:30 AM open with equal capital deployment.
        
        Strategy: Deploy equal capital across all qualifying trades.
        Position size = (Account Cash / Number of candidates) capped at 0.3% ADV.
        
        Returns list of successfully entered positions.
        """
        equity = self.get_account_equity()
        if equity <= 0:
            logger.error("Cannot enter positions: no equity")
            return []
        
        if not candidates:
            return []
        
        # Get actual cash available for trading
        url = f"{self.base_url}/v2/account"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            account_data = response.json()
            cash = float(account_data.get("cash", equity))  # Use cash, not equity
        except:
            cash = equity  # Fallback to equity if cash fetch fails
        
        # Limit to max positions (user set to 100)
        num_positions = min(len(candidates), config.MAX_POSITIONS)
        selected_candidates = candidates[:num_positions]
        
        # Equal capital deployment: cash / number of positions
        target_per_position = cash / num_positions if num_positions > 0 else 0
        
        logger.info(f"Portfolio plan: {num_positions} positions, ${target_per_position:,.2f} per position (from ${cash:,.2f} cash)")
        
        entered = []
        total_allocated = 0.0
        
        for candidate in selected_candidates:
            symbol = candidate.symbol
            expected_price = candidate.open_price
            
            if target_per_position <= 0:
                logger.debug(f"Skipping {symbol}: zero target allocation")
                continue
            
            # Calculate final size with liquidity cap (max 0.3% of ADV)
            quantity = self.calculate_position_size(
                target_per_position,
                candidate.adv_estimate,
                expected_price
            )
            
            if quantity <= 0:
                logger.warning(f"Skipping {symbol}: liquidity cap prevents allocation")
                continue
            
            # Submit market order
            order_id = self._submit_market_order(symbol, quantity, "buy")
            
            if not order_id:
                continue
            
            # Poll for actual fill price
            fill = self.get_order_fill(order_id, max_wait=30)
            
            if not fill:
                logger.error(f"Failed to get fill for {symbol} order {order_id}")
                continue
            
            actual_price = fill["filled_avg_price"]
            actual_qty = int(fill["filled_qty"])
            
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
            
            slippage = ((actual_price / expected_price) - 1) * 100 if expected_price > 0 else 0
            logger.info(f"ENTER {symbol}: {actual_qty} shares @ {actual_price:.2f} (${actual_allocated:,.2f}, slippage: {slippage:+.2f}%, gap: {candidate.gap_pct:.1f}%) [VIX={vix_level:.1f}]")
        
        # Log summary
        logger.info(f"Entry complete: {len(entered)} positions, total allocated=${total_allocated:,.2f} ({total_allocated/cash*100:.1f}% of cash)")
        
        return entered

    def _submit_market_order(
        self, symbol: str, quantity: int, side: str
    ) -> Optional[str]:
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
        """Update position state with current prices. Returns dict of symbol->current_price."""
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

            # Store current price in position and return dict
            position.current_price = current_price
            current_prices[symbol] = current_price

            # Update peak price
            if current_price > position.peak_price:
                position.peak_price = current_price

            # Check trailing stop activation
            entry_price = position.entry_price
            gain_pct = (current_price - entry_price) / entry_price

            if gain_pct >= config.TRAILING_STOP_ACTIVATION:
                position.is_trailing_active = True

            # Update trailing stop level
            if position.is_trailing_active:
                trail_level = position.peak_price * (1 - config.TRAILING_STOP_PCT)
                position.trailing_stop_price = max(position.trailing_stop_price, trail_level)

        return current_prices

    def check_exits(self, current_time: time, vix_level: float, current_prices: Dict[str, float]) -> List[str]:
        """
        Check if any positions should be exited based on:
        1. Trailing stop hits (middle VIX regime only) - checked first for clean precedence
        2. Time-based exits (all VIX regimes)

        For timed exits: distribute across 3 minutes (minute before, minute of, minute after target)
        to help with afternoon liquidity.
        """
        exited = []

        # Determine exit time based on VIX (middle now has time exit too)
        if vix_level < config.VIX_LOW_THRESHOLD:
            target_exit = datetime.strptime(config.EXIT_TIME_LOW_VIX, "%H:%M").time()
        elif vix_level > config.VIX_HIGH_THRESHOLD:
            target_exit = datetime.strptime(config.EXIT_TIME_HIGH_VIX, "%H:%M").time()
        else:
            # Middle regime: 3:30 PM exit
            target_exit = datetime.strptime(config.EXIT_TIME_MIDDLE_VIX, "%H:%M").time()

        # Calculate distributed exit window: 1 minute before to 1 minute after
        target_dt = datetime.combine(datetime.today(), target_exit)
        exit_window_start = (target_dt - timedelta(minutes=1)).time()
        exit_window_end = (target_dt + timedelta(minutes=1)).time()

        for symbol, position in list(self.positions.items()):
            should_exit = False
            exit_reason = ""

            # PRIORITY 1: Trailing stop exit (middle VIX regime only)
            # This takes precedence over time exits in middle regime
            if config.VIX_LOW_THRESHOLD <= vix_level <= config.VIX_HIGH_THRESHOLD:
                if position.is_trailing_active and position.trailing_stop_price > 0:
                    current_price = current_prices.get(symbol, position.current_price)
                    if current_price and current_price <= position.trailing_stop_price:
                        should_exit = True
                        exit_reason = f"trailing_stop ({current_price:.2f} <= {position.trailing_stop_price:.2f})"

            # PRIORITY 2: Time-based exit (all regimes, but only if not already exiting via trailing stop)
            # Distributed across 3-minute window to help with liquidity
            if not should_exit:
                if exit_window_start <= current_time <= exit_window_end:
                    # Check if we've already scheduled this position for exit
                    if symbol not in self.timed_exit_scheduled:
                        # Schedule exit distributed over the window
                        # Use deterministic staggering based on symbol characters (not hash)
                        stagger_seconds = sum(ord(c) for c in symbol) % 120  # 0-119 seconds
                        target_dt = datetime.combine(datetime.today(), exit_window_start)
                        scheduled_time = target_dt + timedelta(seconds=stagger_seconds)
                        self.timed_exit_scheduled[symbol] = scheduled_time
                        logger.debug(f"Scheduled timed exit for {symbol} at {scheduled_time.time()}")

                    # Check if scheduled time has passed (using current_time for consistency)
                    scheduled = self.timed_exit_scheduled.get(symbol)
                    current_dt = datetime.combine(datetime.today(), current_time)
                    if scheduled and current_dt >= scheduled:
                        should_exit = True
                        exit_reason = f"time_exit (VIX={vix_level:.1f}, distributed)"

            if should_exit:
                self._exit_position(symbol, exit_reason)
                exited.append(symbol)
                # Clean up scheduled exit if present
                self.timed_exit_scheduled.pop(symbol, None)

        return exited

    def _exit_position(self, symbol: str, reason: str):
        """Exit a position with market order, using actual fill price"""
        position = self.positions.get(symbol)
        if not position:
            return

        order_id = self._submit_market_order(symbol, position.quantity, "sell")

        if not order_id:
            return

        # Poll for actual fill price
        fill = self.get_order_fill(order_id, max_wait=30)

        if not fill:
            logger.error(f"Failed to get exit fill for {symbol} order {order_id}")
            return

        exit_price = fill["filled_avg_price"]
        filled_qty = fill["filled_qty"]

        pnl = (exit_price - position.entry_price) * filled_qty
        pnl_pct = ((exit_price / position.entry_price) - 1) * 100

        logger.info(f"EXIT {symbol}: {filled_qty} shares @ {exit_price:.2f} (P&L: {pnl:+.2f}, {pnl_pct:+.1f}%) - {reason}")

        del self.positions[symbol]

    def get_position_count(self) -> int:
        """Return number of open positions"""
        return len(self.positions)

    def force_exit_all(self, reason: str = "force"):
        """Force exit all positions immediately"""
        logger.warning(f"Force exiting all positions: {reason}")
        for symbol in list(self.positions.keys()):
            self._exit_position(symbol, reason)
