"""Alpaca ``trade_updates`` websocket — push fills instead of REST polling.

This module runs a background daemon thread that connects to Alpaca's
``/stream`` endpoint and listens for ``trade_updates`` events. Terminal
order events (``fill``, ``partial_fill``, ``canceled``, ``expired``,
``rejected``, ``done_for_day``) are stored in a thread-safe dict keyed
by ``order_id``.

PositionManager checks this dict at the start of every poll iteration of
``get_order_fill`` so a fill seen on the stream short-circuits the REST
polling loop with sub-second latency. If the stream is down the existing
REST polling continues to work — this is a side-channel, not a replacement.

Endpoint format (JSON, NOT msgpack — that is only for market data):

    auth      → {"action":"authenticate","data":{"key_id":..,"secret_key":..}}
    subscribe → {"action":"listen","data":{"streams":["trade_updates"]}}
    message   → {"stream":"trade_updates",
                 "data":{"event":"fill","order":{"id":"..",
                                                  "status":"filled",
                                                  "filled_qty":"100",
                                                  "filled_avg_price":"5.43"}}}
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Order events that mean "this order will not change again"
_TERMINAL_EVENTS = {"fill", "canceled", "expired", "rejected", "done_for_day"}


class FillStream:
    """Background websocket subscriber for Alpaca ``trade_updates``.

    Thread-safe. ``get_terminal_event`` and ``wait_for_terminal_event``
    are the only methods callers need.
    """

    def __init__(self, base_url: str, api_key: str, secret_key: str):
        # Convert REST base URL to wss:// /stream URL.
        wss = base_url.replace("https://", "wss://").rstrip("/") + "/stream"
        self._ws_url = wss
        self._api_key = api_key
        self._secret_key = secret_key

        self._events: Dict[str, dict] = {}     # order_id -> latest snapshot
        self._terminals: Dict[str, dict] = {}  # order_id -> terminal event
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._stop = threading.Event()
        self._connected = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._connected.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="alpaca-fill-stream",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"FillStream: thread started, url={self._ws_url}")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to stop and join briefly."""
        self._stop.set()
        # Wake any waiters
        with self._cv:
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("FillStream: stopped")

    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ────────────────────────────────────────────────────────
    # Public access
    # ────────────────────────────────────────────────────────

    def get_terminal_event(self, order_id: str) -> Optional[dict]:
        """Return cached terminal event for ``order_id`` or None."""
        with self._lock:
            return self._terminals.get(order_id)

    def wait_for_terminal_event(self, order_id: str, timeout: float) -> Optional[dict]:
        """Block up to ``timeout`` seconds for a terminal event for ``order_id``.

        Returns the event dict (with ``status``, ``filled_qty``,
        ``filled_avg_price``) or None on timeout.
        """
        deadline = None
        with self._cv:
            existing = self._terminals.get(order_id)
            if existing is not None:
                return existing
            remaining = timeout
            import time as _t
            deadline = _t.monotonic() + timeout
            while remaining > 0 and not self._stop.is_set():
                self._cv.wait(timeout=remaining)
                evt = self._terminals.get(order_id)
                if evt is not None:
                    return evt
                remaining = deadline - _t.monotonic()
        return None

    # ────────────────────────────────────────────────────────
    # Background thread
    # ────────────────────────────────────────────────────────

    def _thread_main(self) -> None:
        """Run a private asyncio loop forever, with auto-reconnect on failure."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run())
        except Exception:
            logger.exception("FillStream: unhandled exception in thread")

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._connect_once()
                backoff = 1.0  # reset after a clean connect
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected.clear()
                logger.warning(f"FillStream: connection error ({e!r}); retry in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _connect_once(self) -> None:
        # ``websockets`` is already a dep (used by alpaca-py StockDataStream)
        import websockets

        async with websockets.connect(
            self._ws_url,
            ping_interval=30,
            ping_timeout=15,
            close_timeout=5,
        ) as ws:
            # Authenticate
            await ws.send(json.dumps({
                "action": "authenticate",
                "data": {"key_id": self._api_key, "secret_key": self._secret_key},
            }))
            auth_resp = await asyncio.wait_for(ws.recv(), timeout=10)
            try:
                auth_data = json.loads(auth_resp)
            except (ValueError, TypeError):
                raise RuntimeError(f"FillStream: non-JSON auth response: {auth_resp!r}")
            if auth_data.get("data", {}).get("status") != "authorized":
                raise RuntimeError(f"FillStream: auth failed: {auth_data}")

            # Subscribe
            await ws.send(json.dumps({
                "action": "listen",
                "data": {"streams": ["trade_updates"]},
            }))
            self._connected.set()
            logger.info("FillStream: authenticated and listening on trade_updates")

            # Read forever
            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                except asyncio.TimeoutError:
                    # No traffic for 60s is normal at night; the ping_interval
                    # keeps the connection alive on its own.
                    continue
                self._handle_message(raw)

    def _handle_message(self, raw) -> None:
        try:
            msg = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
        except (ValueError, TypeError):
            logger.debug(f"FillStream: non-JSON frame: {raw!r}")
            return

        if not isinstance(msg, dict):
            return
        if msg.get("stream") != "trade_updates":
            return

        data = msg.get("data") or {}
        event = data.get("event")
        order = data.get("order") or {}
        order_id = order.get("id")
        if not order_id:
            return

        # Build a normalized event payload that mirrors what the REST poller
        # would return so the rest of the bot doesn't care which path it came from.
        try:
            filled_qty = float(order.get("filled_qty") or 0)
        except (TypeError, ValueError):
            filled_qty = 0.0
        filled_avg_price_raw = order.get("filled_avg_price")
        try:
            filled_avg_price = float(filled_avg_price_raw) if filled_avg_price_raw else None
        except (TypeError, ValueError):
            filled_avg_price = None

        normalized = {
            "order_id": order_id,
            "event": event,
            "status": order.get("status"),
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg_price,
        }

        with self._cv:
            self._events[order_id] = normalized
            if event in _TERMINAL_EVENTS:
                # Only replace an existing terminal if the new one carries
                # strictly more fill information — guards against out-of-order
                # event delivery for the same order id.
                prev = self._terminals.get(order_id)
                if prev is None or normalized["filled_qty"] >= prev.get("filled_qty", 0):
                    self._terminals[order_id] = normalized
                self._cv.notify_all()
                logger.info(
                    f"FillStream: terminal {event} for {order_id} "
                    f"qty={normalized['filled_qty']} px={normalized['filled_avg_price']}"
                )
