"""Dataclasses representing core runtime entities."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from zoneinfo import ZoneInfo


logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")


@dataclass
class Candidate:
    symbol: str
    price: float
    prev_close: float
    pm_last: float
    pm_high: float
    pm_volume: float
    avg_vol_30d: float
    float_shares: float

    gap_pct: float
    pm_vol_float: float
    relvol: float
    score: float
    
    # Liquidity metric for dynamic position sizing (set during entry loop)
    liq_5m_dollar: float = 0.0


@dataclass
class PositionState:
    symbol: str
    entry_time: datetime
    entry_price: float
    qty: float

    stop_price: float
    peak_price: float

    r_stop_pct: float  # initial stop% used for sizing
    trail_pct: Optional[float] = None
    breakeven_set: bool = False
    trail_active: bool = False

    entry_order_id: Optional[str] = None
    entry_client_order_id: Optional[str] = None

    spread_bad_count: int = 0
    exit_pending: bool = False           # True while an exit order is in flight
    exit_order_id: Optional[str] = None
    exit_client_order_id: Optional[str] = None
    exit_submitted_ts: Optional[float] = None  # monotonic seconds at submission (for age calculations)
    exit_attempts: int = 0
    exit_reason: str = ""

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_r: Optional[float] = None


@dataclass
class PendingEntryState:
    symbol: str
    client_order_id: str
    submitted_ts: float        # epoch seconds
    attempts: int = 1
    stop_pct: float = 0.0      # needed to open PositionState on adoption
    intended_qty: float = 0.0      # intended qty at submission
    intended_price: float = 0.0
    cancel_requested_ts: float = 0.0   # epoch seconds of last cancel attempt (0 = never)
    cancel_attempts: int = 0           # number of cancel attempts made


def pending_entry_to_dict(p: PendingEntryState) -> Dict[str, Any]:
    return {
        "symbol": p.symbol,
        "client_order_id": p.client_order_id,
        "submitted_ts": p.submitted_ts,
        "attempts": p.attempts,
        "stop_pct": p.stop_pct,
        "intended_qty": p.intended_qty,
        "intended_price": p.intended_price,
        "cancel_requested_ts": p.cancel_requested_ts,
        "cancel_attempts": p.cancel_attempts,
    }


def pending_entry_from_dict(payload: Dict[str, Any]) -> PendingEntryState:
    return PendingEntryState(
        symbol=payload["symbol"],
        client_order_id=payload["client_order_id"],
        submitted_ts=float(payload["submitted_ts"]),
        attempts=int(payload.get("attempts", 1)),
        stop_pct=float(payload.get("stop_pct", 0.0)),
        intended_qty=float(payload.get("intended_qty", 0.0)),
        intended_price=float(payload.get("intended_price", 0.0)),
        cancel_requested_ts=float(payload.get("cancel_requested_ts", 0.0)),
        cancel_attempts=int(payload.get("cancel_attempts", 0)),
    )


def _ensure_market_tz(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=MARKET_TZ)
    return dt.astimezone(MARKET_TZ)


def position_state_to_dict(state: PositionState) -> Dict[str, Any]:
    entry_time = _ensure_market_tz(state.entry_time)
    exit_time = _ensure_market_tz(state.exit_time)
    return {
        "symbol": state.symbol,
        "entry_time": entry_time.isoformat() if entry_time else None,
        "entry_price": state.entry_price,
        "qty": state.qty,
        "stop_price": state.stop_price,
        "peak_price": state.peak_price,
        "r_stop_pct": state.r_stop_pct,
        "trail_pct": state.trail_pct,
        "breakeven_set": state.breakeven_set,
        "trail_active": state.trail_active,
        "entry_order_id": state.entry_order_id,
        "entry_client_order_id": state.entry_client_order_id,
        "spread_bad_count": state.spread_bad_count,
        "exit_pending": state.exit_pending,
        "exit_order_id": state.exit_order_id,
        "exit_client_order_id": state.exit_client_order_id,
        "exit_submitted_ts": state.exit_submitted_ts,
        "exit_attempts": state.exit_attempts,
        "exit_reason": state.exit_reason,
        "exit_time": exit_time.isoformat() if exit_time else None,
        "exit_price": state.exit_price,
        "realized_r": state.realized_r,
    }


def position_state_from_dict(payload: Dict[str, Any]) -> Optional[PositionState]:
    try:
        entry_raw = payload["entry_time"]
        if not entry_raw:
            raise ValueError("missing entry_time")
        entry_time = _ensure_market_tz(datetime.fromisoformat(entry_raw))

        exit_raw = payload.get("exit_time")
        exit_time = _ensure_market_tz(datetime.fromisoformat(exit_raw)) if exit_raw else None

        return PositionState(
            symbol=str(payload["symbol"]),
            entry_time=entry_time,
            entry_price=float(payload["entry_price"]),
            qty=float(payload["qty"]),
            stop_price=float(payload["stop_price"]),
            peak_price=float(payload["peak_price"]),
            r_stop_pct=float(payload["r_stop_pct"]),
            trail_pct=(
                float(payload["trail_pct"])
                if payload.get("trail_pct") is not None
                else None
            ),
            breakeven_set=bool(payload.get("breakeven_set", False)),
            trail_active=bool(payload.get("trail_active", False)),
            entry_order_id=payload.get("entry_order_id") or None,
            entry_client_order_id=payload.get("entry_client_order_id") or None,
            spread_bad_count=int(payload.get("spread_bad_count", 0)),
            exit_pending=bool(payload.get("exit_pending", False)),
            exit_order_id=payload.get("exit_order_id") or None,
            exit_client_order_id=payload.get("exit_client_order_id") or None,
            exit_submitted_ts=(
                float(payload["exit_submitted_ts"])
                if payload.get("exit_submitted_ts") is not None
                else None
            ),
            exit_attempts=int(payload.get("exit_attempts", 0)),
            exit_reason=str(payload.get("exit_reason", "")),
            exit_time=exit_time,
            exit_price=(
                float(payload["exit_price"])
                if payload.get("exit_price") is not None
                else None
            ),
            realized_r=(
                float(payload["realized_r"])
                if payload.get("realized_r") is not None
                else None
            ),
        )
    except Exception as exc:
        symbol = payload.get("symbol", "UNKNOWN")
        logger.error("Failed to deserialize position state for %s: %s", symbol, exc)
        return None
