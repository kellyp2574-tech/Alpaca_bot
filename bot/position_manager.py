"""Position sizing helpers and live exit management."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .morning_main import SessionStats

from .clock import market_now
from .morning_config import Config
from .execution import ExecutionClient, FillResult
from .state_manager import StateStore
from .storage import PositionState
from alpaca.trading.enums import OrderSide

_SESSION_DATE: Optional[str] = None


def _session_date() -> str:
    """Return today's date string, updating if day has changed."""
    global _SESSION_DATE
    today = market_now().strftime("%Y%m%d")
    if _SESSION_DATE != today:
        _SESSION_DATE = today
    return _SESSION_DATE


_CLIENT_ID_MAX_LEN: int = 48  # Alpaca client_order_id maximum length


def _norm_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _entry_client_id(symbol: str, attempt: int = 1) -> str:
    raw = f"ENTRY:{_norm_symbol(symbol)}:{_session_date()}:{attempt}"
    if len(raw) > _CLIENT_ID_MAX_LEN:
        raise ValueError(f"client_order_id too long ({len(raw)}): {raw!r}")
    return raw


def _exit_client_id(symbol: str, attempt: int) -> str:
    raw = f"EXIT:{_norm_symbol(symbol)}:{_session_date()}:{attempt}"
    if len(raw) > _CLIENT_ID_MAX_LEN:
        raise ValueError(f"client_order_id too long ({len(raw)}): {raw!r}")
    return raw

logger = logging.getLogger(__name__)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def initial_stop_pct(cfg: Config, atr: float, entry: float) -> float:
    if entry <= 0 or atr <= 0:
        return cfg.stop_max_pct
    raw = cfg.stop_atr_mult * (atr / entry)
    return clamp(raw, cfg.stop_min_pct, cfg.stop_max_pct)


def calc_qty(
    account_equity: float, risk_pct: float, entry: float, stop_pct: float
) -> float:
    risk_dollars = account_equity * risk_pct
    stop_dollars = entry * stop_pct
    if stop_dollars <= 0:
        return 0.0
    shares = risk_dollars / stop_dollars
    return max(0.0, shares)


class PositionManager:
    def __init__(
        self,
        cfg: Config,
        execution: ExecutionClient,
        risk_manager,
        *,
        state_store: Optional[StateStore] = None,
    ) -> None:
        self.cfg = cfg
        self.execution = execution
        self.risk_manager = risk_manager
        self.positions: Dict[str, PositionState] = {}
        self.state_store = state_store
        self.stats: Optional[SessionStats] = None

    def load_states(self, states: Dict[str, PositionState]) -> None:
        self.positions = dict(states)
        self._persist()

    @property
    def open_count(self) -> int:
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def open_position(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        stop_pct: float,
        *,
        entry_time: Optional[datetime] = None,
        entry_order_id: Optional[str] = None,
        entry_client_order_id: Optional[str] = None,
    ) -> PositionState:
        entry_time = entry_time or market_now()
        stop_price = entry_price * (1 - stop_pct)
        state = PositionState(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            qty=qty,
            stop_price=stop_price,
            peak_price=entry_price,
            r_stop_pct=stop_pct,
            entry_order_id=entry_order_id,
            entry_client_order_id=entry_client_order_id,
        )
        self.positions[symbol] = state
        logger.info(
            "Opened position %s qty=%.4f entry=%.2f stop=%.2f (%.2f%%)",
            symbol,
            qty,
            entry_price,
            stop_price,
            stop_pct * 100,
        )
        self.risk_manager.on_new_trade()
        self._persist()
        return state

    def on_bar(self, symbol: str, bar, *, now: Optional[datetime] = None) -> None:
        state = self.positions.get(symbol)
        if not state:
            return

        now = now or market_now()

        # If an exit is in flight, run reconcile and skip new exit intents,
        # but still update defensive state (trail, peak) so it stays current.
        if state.exit_pending:
            self._reconcile_pending_exit(symbol, state, now)
            state.peak_price = max(state.peak_price, bar.h)
            self._update_trail(state)
            self._persist()
            return

        state.peak_price = max(state.peak_price, bar.h)
        entry = state.entry_price
        price = bar.c

        self._update_breakeven(symbol, state)
        self._update_trail(state)
        
        # Stop loss is handled by main.py _check_position_exits using quotes
        # Bar-based stop disabled to avoid duplicate/conflicting triggers
        # if bar.l <= state.stop_price:
        #     self._exit(symbol, state, state.stop_price, now, reason="stop")
        #     return

        self._persist()

    def _update_breakeven(self, symbol: str, state: PositionState) -> None:
        if not state.breakeven_set and state.peak_price >= state.entry_price * (
            1 + self.cfg.breakeven_at_pct
        ):
            state.stop_price = max(state.stop_price, state.entry_price)
            state.breakeven_set = True
            logger.info("%s stop moved to breakeven", symbol)

    def _update_trail(self, state: PositionState) -> None:
        entry = state.entry_price
        
        # Gap strategy: activate at take_profit_pct, trail at trail_pct
        if not state.trail_active and state.peak_price >= entry * (
            1 + self.cfg.take_profit_pct
        ):
            state.trail_active = True
            state.trail_pct = self.cfg.trail_pct
            logger.info("%s trail activated at %.2f%%", state.symbol, state.trail_pct * 100)

        if state.trail_active:
            trail_pct = state.trail_pct if state.trail_pct else self.cfg.trail_pct
            trail_stop = state.peak_price * (1 - trail_pct)
            state.stop_price = max(state.stop_price, trail_stop)

    def exit_position(
        self, symbol: str, price: float, now: datetime, *, reason: str
    ) -> None:
        """Public interface to exit a single named position."""
        state = self.positions.get(symbol)
        if state:
            self._exit(symbol, state, price, now, reason=reason)

    def force_exit_all(
        self, price_lookup: Dict[str, float], *, reason: str = "hard_exit"
    ) -> None:
        for symbol, state in list(self.positions.items()):
            # Use current bid price from lookup, fallback to reference price, not peak
            price = price_lookup.get(symbol)
            if not price or price <= 0:
                price = self.execution._reference_price(symbol, OrderSide.SELL, state.entry_price)
            self._exit(symbol, state, price, market_now(), reason=reason)

    def reconcile_pending_exits_time_based(self, max_wait_seconds: float = 30.0) -> int:
        """Time-based reconciliation of pending exits (not bar-driven). Returns remaining pending count."""
        start_time = time.monotonic()
        
        while time.monotonic() - start_time < max_wait_seconds:
            pending_count = 0
            for symbol, state in list(self.positions.items()):
                if state.exit_pending:
                    pending_count += 1
                    # Force reconcile without waiting for bars
                    self._reconcile_pending_exit_time_based(symbol, state, market_now())
            
            if pending_count == 0:
                logger.info("All exits reconciled successfully")
                return 0
            
            logger.info(f"Waiting for {pending_count} exits to reconcile...")
            time.sleep(1.0)  # Check every second
        
        remaining_pending = sum(1 for state in self.positions.values() if state.exit_pending)
        logger.warning(f"{remaining_pending} exits still pending after {max_wait_seconds}s")
        return remaining_pending

    def _reconcile_pending_exit_time_based(
        self, symbol: str, state: PositionState, now: datetime
    ) -> None:
        """Time-based reconcile: called every second regardless of bar flow."""
        if not state.exit_submitted_ts:
            # Shouldn't happen, but clear after timeout to avoid permanent lock
            logger.warning("%s exit_pending with no submitted_ts; clearing", symbol)
            state.exit_pending = False
            return

        age = time.monotonic() - state.exit_submitted_ts
        if age < self.cfg.exit_ack_timeout_seconds:
            return  # still within grace window, wait

        fallback = state.exit_price or state.peak_price

        # Prefer order_id lookup; fall back to client_order_id search
        fill: Optional[FillResult] = None
        if state.exit_order_id:
            logger.info(
                "Time-based reconcile exit for %s via order_id %s (age %.0fs)",
                symbol, state.exit_order_id, age,
            )
            try:
                fill = self.execution.poll_order_fill(
                    state.exit_order_id, fallback_price=fallback
                )
            except Exception:
                logger.exception("Time-based reconcile poll failed for %s", symbol)
                return
        elif state.exit_client_order_id:
            logger.info(
                "Time-based reconcile exit for %s via client_order_id %s (order_id lost)",
                symbol, state.exit_client_order_id,
            )
            fill = self.execution.find_order_by_client_id(state.exit_client_order_id)

        if fill is None:
            # Transient error - keep pending and retry next cycle
            logger.warning("%s time-based reconcile transient error; keeping pending for retry", symbol)
            return

        self._apply_fill_result(symbol, state, fill, now)

    # ------------------------------------------------------------------

    def _reconcile_pending_exit(
        self, symbol: str, state: PositionState, now: datetime
    ) -> None:
        """Slow-path reconcile: called each bar while exit_pending.
        Waits exit_ack_timeout_seconds before querying broker.
        Falls back to client_order_id search if order_id was lost.
        """
        if not state.exit_submitted_ts:
            # Shouldn't happen, but clear after timeout to avoid permanent lock
            logger.warning("%s exit_pending with no submitted_ts; clearing", symbol)
            state.exit_pending = False
            return

        age = time.monotonic() - state.exit_submitted_ts  # Use monotonic for consistency
        if age < self.cfg.exit_ack_timeout_seconds:
            return  # still within grace window, wait

        fallback = state.exit_price or state.peak_price

        # Prefer order_id lookup; fall back to client_order_id search
        fill: Optional[FillResult] = None
        if state.exit_order_id:
            logger.info(
                "Reconciling exit for %s via order_id %s (age %.0fs)",
                symbol, state.exit_order_id, age,
            )
            try:
                fill = self.execution.poll_order_fill(
                    state.exit_order_id, fallback_price=fallback
                )
            except Exception:
                logger.exception("Reconcile poll failed for %s", symbol)
                return
        elif state.exit_client_order_id:
            logger.info(
                "Reconciling exit for %s via client_order_id %s (order_id lost)",
                symbol, state.exit_client_order_id,
            )
            fill = self.execution.find_order_by_client_id(state.exit_client_order_id)

        if fill is None:
            # Transient error - keep pending and retry next bar
            logger.warning("%s reconcile transient error; keeping pending for retry", symbol)
            return

        self._apply_fill_result(symbol, state, fill, now)

    def _apply_fill_result(
        self, symbol: str, state: PositionState, fill: FillResult, now: datetime
    ) -> None:
        """Dispatch a FillResult to the appropriate handler. Used by both _exit and reconcile."""
        if fill.status in {"filled", "dry_run"}:
            self._record_fill(symbol, state, fill, now)
        elif fill.status == "partial":
            logger.warning(
                "%s partial fill on exit: %.4f/%.4f shares @ %.2f",
                symbol, fill.filled_qty, state.qty, fill.avg_price,
            )
            if self.stats is not None:
                latency = time.monotonic() - state.exit_submitted_ts if state.exit_submitted_ts else 0.0
                self.stats.record_exit(
                    status="partial",
                    latency=latency,
                    decision_price=state.exit_price or fill.avg_price,
                    fill_price=fill.avg_price,
                )

            remaining_qty = max(state.qty - fill.filled_qty, 0.0)
            broker_qty = None
            if hasattr(self.execution, "_get_broker_qty"):
                try:
                    broker_qty = self.execution._get_broker_qty(symbol)
                except Exception as exc:
                    logger.warning(
                        "Failed to query broker qty for %s after partial fill: %s",
                        symbol,
                        exc,
                    )
            if broker_qty is not None:
                remaining_qty = max(float(broker_qty), 0.0)

            state.qty = remaining_qty
            state.exit_pending = False
            state.exit_order_id = None
            state.exit_client_order_id = None
            state.exit_submitted_ts = None
            if state.qty <= 0.0:
                self.positions.pop(symbol, None)
            self._persist()
            return
        elif fill.status == "unfilled":
            logger.warning(
                "Exit order for %s was unfilled; position remains open", symbol
            )
            if self.stats is not None:
                self.stats.record_exit(
                    status="unfilled", latency=0.0,
                    decision_price=0.0, fill_price=0.0,
                )
            state.exit_pending = False
            state.exit_order_id = None
            state.exit_client_order_id = None
            state.exit_submitted_ts = None
            self._persist()
        elif fill.status == "unknown":
            logger.warning("%s exit order still unknown after reconcile", symbol)
            if self.stats is not None:
                self.stats.record_exit(
                    status="unknown", latency=0.0,
                    decision_price=0.0, fill_price=0.0,
                )
            # Leave pending; will retry next bar

    def _record_fill(
        self, symbol: str, state: PositionState, fill: FillResult, now: datetime
    ) -> None:
        """Finalize a confirmed fill: compute R, notify risk manager, remove position."""
        decision_price = state.exit_price or state.peak_price  # provisional price set at _exit time
        price = fill.avg_price if fill.avg_price > 0 else decision_price
        if state.r_stop_pct > 0:
            state.realized_r = (price - state.entry_price) / (
                state.entry_price * state.r_stop_pct
            )
        else:
            state.realized_r = 0.0
        state.exit_time = now
        state.exit_price = price
        self.risk_manager.on_trade_closed(state.realized_r or 0.0)
        logger.info(
            "EXIT %s qty=%.4f @ %.2f reason=%s R=%.2f order=%s",
            symbol, fill.filled_qty, price, state.exit_reason,
            state.realized_r or 0.0, fill.order_id,
        )
        if self.stats is not None:
            latency = (
                time.monotonic() - state.exit_submitted_ts
                if state.exit_submitted_ts else 0.0
            )
            self.stats.record_exit(
                status=fill.status,
                latency=latency,
                decision_price=decision_price,
                fill_price=price,
            )
        self.positions.pop(symbol, None)
        self._persist()

    def _exit(
        self,
        symbol: str,
        state: PositionState,
        price: float,
        now: datetime,
        *,
        reason: str,
    ) -> None:
        """Exit position using IOC (Immediate or Cancel) to prevent hanging orders.
        
        IOC ensures immediate fill or cancel - no resting exit orders.
        If unfilled, will retry up to 3 times with more aggressive pricing.
        After 3 failed attempts, will emergency flatten the position.
        """
        if state.exit_pending:
            logger.debug("Exit already in flight for %s, skipping duplicate", symbol)
            return

        # Use iterative retry instead of recursion to prevent stack overflow
        max_attempts = 3
        current_price = price
        
        for attempt in range(1, max_attempts + 1):
            state.exit_attempts = attempt
            client_id = _exit_client_id(symbol, attempt)
            state.exit_pending = True
            state.exit_submitted_ts = time.monotonic()  # Use monotonic for consistent time basis
            state.exit_reason = reason
            state.exit_client_order_id = client_id
            state.exit_price = current_price  # provisional; overwritten by actual fill
            self._persist()

            qty = state.qty
            fill = self.execution.place_exit(symbol, qty, current_price, client_order_id=client_id)
            state.exit_order_id = fill.order_id
            self._persist()

            self._apply_fill_result(symbol, state, fill, now)
            
            # Check if we're done
            if fill.status in {"filled", "dry_run"}:
                logger.info(f"Exit {symbol} completed on attempt {attempt}: {fill.status}")
                return
            elif fill.status == "partial":
                if state.qty <= 0.0:
                    logger.info(f"Exit {symbol} fully closed via partial fills on attempt {attempt}")
                    return
                if attempt < max_attempts:
                    aggressiveness_factors = [0.995, 0.99, 0.985]
                    aggressiveness = aggressiveness_factors[min(attempt - 1, len(aggressiveness_factors) - 1)]
                    new_price = max(state.exit_price or current_price, 0.0) * aggressiveness
                    current_price = new_price if new_price > 0 else current_price
                    logger.info(
                        "Partial exit for %s left %.4f shares; retrying with price %.2f",
                        symbol,
                        state.qty,
                        current_price,
                    )
                    continue
                else:
                    logger.error(f"Partial exit could not close {symbol} after {attempt} attempts - escalating to broker close_position")
                    fill = FillResult(order_id=None, filled_qty=0.0, avg_price=current_price, status="unfilled")
                    # fall through to escalation logic below

            if fill.status == "unfilled" or (fill.status == "partial" and attempt >= max_attempts):
                if attempt < max_attempts:
                    logger.warning(f"IOC exit unfilled for {symbol}, attempt {attempt} - retrying with more aggressive price")
                    # Retry with more aggressive pricing for SELL orders (lower limit = more marketable)
                    # Use progressive aggressiveness: 0.995, 0.99, 0.985
                    aggressiveness_factors = [0.995, 0.99, 0.985]
                    aggressiveness = aggressiveness_factors[min(attempt - 1, len(aggressiveness_factors) - 1)]
                    current_price = price * aggressiveness
                    logger.info(f"Retry {attempt}: price {price:.2f} -> {current_price:.2f} (factor {aggressiveness})")
                    continue  # Next iteration with new price
                else:
                    logger.error(f"IOC exit failed after {attempt} attempts for {symbol} - escalating to broker close_position")
                    # Emergency: escalate to broker close_position (market order) to guarantee flat
                    try:
                        if hasattr(self.execution, 'client') and self.execution.client:
                            logger.critical(f"Escalating {symbol} to broker close_position (market order)")
                            order = self.execution.client.close_position(symbol)
                            state.exit_reason = "emergency_market_close"
                            state.exit_submitted_ts = time.monotonic()
                            state.exit_pending = True
                            state.exit_order_id = getattr(order, "id", state.exit_order_id)
                            if getattr(order, "client_order_id", None):
                                state.exit_client_order_id = order.client_order_id
                            state.exit_price = current_price  # provisional until fill confirmed
                            state.exit_time = None
                            self._persist()
                            logger.info(f"Emergency market close submitted for {symbol}")
                        else:
                            logger.critical(f"Cannot escalate {symbol} - no broker client available")
                            # Keep position tracked with alarm
                            state.exit_pending = False
                            state.exit_reason = "emergency_exit_failed_no_client"
                            self._persist()
                    except Exception as e:
                        logger.critical(f"Emergency close_position failed for {symbol}: {e}")
                        # Keep position tracked with alarm - DO NOT remove from local state
                        state.exit_pending = False
                        state.exit_reason = "emergency_exit_failed_exception"
                        self._persist()
                    return
            else:
                logger.warning(f"Exit {symbol} status {fill.status} - keeping pending for reconcile")
                return  # Let time-based reconcile handle unknown status

    def _persist(self) -> None:
        if not self.state_store:
            return
        self.state_store.save_positions(self.positions)
