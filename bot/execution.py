"""Order execution helpers (marketable limits + synthetic stops)."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import LimitOrderRequest
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


load_dotenv()


def _is_not_found(exc: Exception) -> bool:
    """Return True if *exc* represents a definitive 404 / resource-not-found response.

    Alpaca-py raises ``alpaca.common.exceptions.APIError`` for HTTP errors.
    We check the status_code attribute first, then fall back to inspecting the
    string representation so the helper stays robust across SDK versions.
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        return int(status) == 404
    # Fallback: requests HTTPError or similar
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if code is not None:
            return int(code) == 404
    # Last resort: string match
    return "404" in str(exc) or "not found" in str(exc).lower()


@dataclass
class FillResult:
    order_id: Optional[str]
    filled_qty: float
    avg_price: float
    status: str  # 'filled', 'partial', 'unfilled', 'unknown', 'dry_run'


@dataclass
class ExecutionConfig:
    buy_slippage_pct: float = 0.002  # 0.2%
    sell_slippage_pct: float = 0.005  # 0.5%
    min_price: float = 0.01
    fill_poll_interval_s: float = 0.1   # seconds between fill status polls
    fill_poll_max_s: float = 2.0        # total time to poll before giving up (slow reconcile)
    quick_poll_max_s: float = 1.0       # fast first-pass poll window right after submit


class ExecutionClient:
    """Thin wrapper around Alpaca TradingClient for marketable orders."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        *,
        paper: Optional[bool] = None,
        cfg: Optional[ExecutionConfig] = None,
        quote_provider: Optional[Callable[[str], Any]] = None,
        dry_run: bool = False,
    ) -> None:
        api_key = api_key or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret_key = secret_key or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        if paper is None:
            paper_env = os.getenv("ALPACA_PAPER")
            if paper_env is not None:
                paper = paper_env.lower() in {"1", "true", "yes", "on"}
            else:
                paper = True
        if not dry_run and (not api_key or not secret_key):
            raise ValueError(
                "ExecutionClient requires Alpaca API credentials (or use dry_run)"
            )

        self.dry_run = dry_run
        self.client = (
            None if dry_run else TradingClient(api_key, secret_key, paper=paper)
        )
        self.cfg = cfg or ExecutionConfig()
        self.quote_provider = quote_provider

    def _reference_price(self, symbol: str, side: OrderSide, fallback: float) -> float:
        if not self.quote_provider:
            return fallback
        try:
            quote = self.quote_provider(symbol)
        except Exception:  # pragma: no cover - network errors
            logger.exception("Failed to fetch quote for %s", symbol)
            return fallback

        if side == OrderSide.BUY:
            ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
            if ask > 0:
                return ask
        else:
            bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
            if bid > 0:
                return bid
        return fallback

    def _marketable_limit(self, symbol: str, side: OrderSide, fallback_price: float) -> float:
        price = self._reference_price(symbol, side, fallback_price)
        if side == OrderSide.BUY:
            slip = self.cfg.buy_slippage_pct
            raw = price * (1 + slip)
            # Ensure we're not too aggressive on low-priced stocks
            # Use max to ensure we don't go below a reasonable marketable level
            raw = max(raw, price + max(0.01, price * 0.001))  # At least 0.01 or 0.1% of price
        else:
            slip = self.cfg.sell_slippage_pct
            raw = price * (1 - slip)
            # Ensure we maintain a marketable discount without going negative
            tick = max(0.01, price * 0.001)
            raw = max(raw, price - tick)
        raw = max(raw, self.cfg.min_price)
        return round(raw, 2)

    def place_entry(
        self, symbol: str, qty: float, last_price: float, *, client_order_id: str
    ) -> FillResult:
        limit_price = self._marketable_limit(symbol, OrderSide.BUY, last_price)
        return self._submit_and_poll(
            symbol, qty, OrderSide.BUY, limit_price,
            client_order_id=client_order_id, poll_max_s=self.cfg.quick_poll_max_s,
        )

    def place_exit(
        self,
        symbol: str,
        qty: float,
        last_price: float,
        *,
        client_order_id: str,
        force_price: bool = False,
    ) -> FillResult:
        # Reduce-only guard: clamp qty to broker-confirmed position size
        broker_qty = self._get_broker_qty(symbol)
        if broker_qty is not None and qty > broker_qty:
            logger.warning(
                "Clamping exit qty for %s from %s to broker qty %.4f",
                symbol, qty, broker_qty,
            )
            qty = float(broker_qty)
        if qty <= 0:
            logger.warning("Exit for %s skipped: broker reports 0 position", symbol)
            return FillResult(order_id=None, filled_qty=0.0, avg_price=0.0, status="unfilled")
        if force_price:
            limit_price = round(max(last_price, self.cfg.min_price), 2)
        else:
            limit_price = self._marketable_limit(symbol, OrderSide.SELL, last_price)
        return self._submit_and_poll(
            symbol, qty, OrderSide.SELL, limit_price,
            client_order_id=client_order_id, poll_max_s=self.cfg.quick_poll_max_s,
        )

    def _get_broker_qty(self, symbol: str) -> Optional[float]:
        """Return the broker-confirmed long qty for symbol.

        Returns:
            float >= 0  — broker holds this many shares (including fractional)
            0           — position definitively absent (404)
            None        — transient error; caller should not treat as absent
        """
        if self.dry_run or self.client is None:
            return None
        try:
            pos = self.client.get_open_position(symbol)
            return max(0.0, float(pos.qty))
        except Exception as exc:
            if _is_not_found(exc):
                return 0
            logger.warning("Transient error fetching broker position for %s: %s", symbol, exc)
            return None

    def is_fractionable(self, symbol: str) -> Optional[bool]:
        """Check if a symbol can be traded fractionally.

        Returns:
            True   — fractional trading allowed
            False  — whole shares only
            None   — transient error; caller should default to whole shares
        """
        if self.dry_run or self.client is None:
            return False
        try:
            asset = self.client.get_asset(symbol)
            return getattr(asset, "fractionable", False)
        except Exception:
            logger.warning("Failed to check fractionable status for %s, returning None", symbol)
            return None

    def find_order_by_client_id(self, client_order_id: str) -> Optional[FillResult]:
        """Search for an order by deterministic client_order_id. Used for crash recovery.

        Returns:
            FillResult  — order found; status reflects its state
            None        — transient error; caller should apply grace window
        """
        if self.dry_run or self.client is None:
            return None
        try:
            order = self.client.get_order_by_client_id(client_order_id)
            return self._order_to_fill_result(order)
        except Exception as exc:
            if _is_not_found(exc):
                return FillResult(order_id=None, filled_qty=0.0, avg_price=0.0, status="unfilled")
            logger.warning("Transient error looking up order %s: %s", client_order_id, exc)
            return None

    def _submit_and_poll(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        limit_price: float,
        *,
        client_order_id: str,
        poll_max_s: float,
    ) -> FillResult:
        if qty <= 0:
            return FillResult(order_id=None, filled_qty=0.0, avg_price=0.0, status="unfilled")

        if self.dry_run:
            logger.info(
                "[DRY] %s %s qty %s @ %.2f client_id=%s",
                side.value, symbol, qty, limit_price, client_order_id,
            )
            return FillResult(order_id=None, filled_qty=float(qty), avg_price=limit_price, status="dry_run")

        # Check if fractional and if symbol supports it
        is_fractional = (qty % 1) != 0
        
        if is_fractional:
            # Check if symbol is fractionable
            fractionable = self.is_fractionable(symbol)
            if fractionable is False:
                # Floor to whole shares if not fractionable
                import math
                original_qty = qty
                qty = math.floor(qty)
                logger.info(
                    "%s not fractionable: %.2f shares → %d shares",
                    symbol, original_qty, qty
                )
                if qty <= 0:
                    return FillResult(order_id=None, filled_qty=0.0, avg_price=0.0, status="unfilled")
                is_fractional = False
            elif fractionable is None:
                # Transient error - default to whole shares for safety
                import math
                qty = math.floor(qty)
                logger.warning(
                    "%s fractionable check failed: defaulting to %d whole shares",
                    symbol, qty
                )
                if qty <= 0:
                    return FillResult(order_id=None, filled_qty=0.0, avg_price=0.0, status="unfilled")
                is_fractional = False
        
        if is_fractional:
            tif = TimeInForce.DAY  # Required for fractional shares
        else:
            tif = TimeInForce.IOC  # Immediate or Cancel for whole shares
        
        order = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            type=OrderType.LIMIT,
            time_in_force=tif,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        resp = self.client.submit_order(order)
        order_id = str(resp.id)
        logger.info(
            "Submitted %s %s qty %s @ %.2f order_id=%s client_id=%s tif=%s",
            side.value, symbol, qty, limit_price, order_id, client_order_id, tif.value,
        )
        return self.poll_order_fill(order_id, fallback_price=limit_price, poll_max_s=poll_max_s)

    def poll_order_fill(
        self, order_id: str, *, fallback_price: float = 0.0, poll_max_s: Optional[float] = None
    ) -> FillResult:
        """Poll order status until terminal or timeout. Returns best available fill info."""
        if poll_max_s is None:
            poll_max_s = self.cfg.fill_poll_max_s
        deadline = time.monotonic() + poll_max_s
        while time.monotonic() < deadline:
            try:
                order = self.client.get_order_by_id(order_id)
            except Exception:
                logger.warning("Transient error polling order %s; retrying", order_id, exc_info=True)
                time.sleep(self.cfg.fill_poll_interval_s)
                continue
            result = self._order_to_fill_result(order, fallback_price=fallback_price)
            if result.status != "unknown":
                return result
            time.sleep(self.cfg.fill_poll_interval_s)

        logger.warning("Order %s still live after %.1fs poll window", order_id, poll_max_s)
        try:
            order = self.client.get_order_by_id(order_id)
            filled_qty = float(order.filled_qty or 0)
            avg_price = float(order.filled_avg_price or fallback_price)
            return FillResult(order_id=order_id, filled_qty=filled_qty, avg_price=avg_price, status="unknown")
        except Exception:
            return FillResult(order_id=order_id, filled_qty=0.0, avg_price=fallback_price, status="unknown")

    @staticmethod
    def _order_to_fill_result(
        order: Any, *, fallback_price: float = 0.0
    ) -> FillResult:
        """Map an Alpaca order object to a FillResult with correct terminal status."""
        order_id = str(order.id)
        status = str(order.status).lower()
        filled_qty = float(order.filled_qty or 0)
        avg_price = float(order.filled_avg_price or fallback_price)

        if status == "filled":
            return FillResult(order_id=order_id, filled_qty=filled_qty, avg_price=avg_price, status="filled")

        if status == "partially_filled":
            return FillResult(order_id=order_id, filled_qty=filled_qty, avg_price=avg_price, status="partial")

        if status in {"canceled", "expired"}:
            # IOC cancel: if any shares filled, that's a partial; otherwise unfilled
            if filled_qty > 0:
                return FillResult(order_id=order_id, filled_qty=filled_qty, avg_price=avg_price, status="partial")
            return FillResult(order_id=order_id, filled_qty=0.0, avg_price=0.0, status="unfilled")

        if status == "rejected":
            return FillResult(order_id=order_id, filled_qty=0.0, avg_price=0.0, status="unfilled")

        # new, accepted, pending_new, held, done_for_day, etc. — not yet terminal
        return FillResult(order_id=order_id, filled_qty=filled_qty, avg_price=avg_price, status="unknown")
