"""P1 Fallback Module — Priority 3 intraday sleeve (Router > V2 > P1)

P1 evaluates when both the primary ETF router AND V2 fallback return NO_TRADE.
- P1 only fires on LIVE_BULLISH_MESSY subtype with XLK > 0
- Entry at 10:15, exit at 15:00
- Vehicle: TQQQ
- Uses 90% ETF bucket (same as router and V2)
- Bracket order with SL=-4% / TP=+5% (per ETF_SL_TP["P1_FALLBACK"]);

Every function takes the orchestrator instance (`bot`) as first argument
and mutates state directly. State ownership stays with the orchestrator.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from bot import config
from bot.universe_builder import filter_execution_ready

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


def _parse_hhmm(value: str) -> dt_time:
    """Parse 'HH:MM' config string into time object."""
    parts = str(value).split(":")
    return dt_time(int(parts[0]), int(parts[1]))


def evaluate_p1_fallback(bot, current_time: dt_time) -> bool:
    """Evaluate P1 fallback entry conditions.
    
    P1 only fires if:
    - Router returned no-trade
    - V2 did not fire (no position taken)
    - Subtype is LIVE_BULLISH_MESSY
    - XLK return > 0
    - Current time is P1_ENTRY_TIME (default 10:15)
    
    Returns True if P1 should fire.
    """
    if not getattr(config, "ENABLE_P1_FALLBACK", False):
        return False
    
    # Check that we're at or after the entry time (Issue 3 follow-up: prevent 10:14)
    entry_t = _parse_hhmm(getattr(config, "P1_ENTRY_TIME", "10:15"))
    if current_time < entry_t:
        return False
    
    # Check that V2 did not fire (no intraday position yet)
    if getattr(bot, "intraday_etf_sleeve_filled", False):
        logger.info("P1: intraday ETF sleeve already filled (V2 or router), skipping P1")
        return False
    
    # Check subtype is LIVE_BULLISH_MESSY
    required_subtype = getattr(config, "P1_REQUIRED_SUBTYPE", "LIVE_BULLISH_MESSY")
    subtype = getattr(bot, "router_no_trade_subtype", None)
    if subtype != required_subtype:
        logger.info(f"P1: subtype={subtype} != required {required_subtype}")
        return False
    
    # Check XLK > 0 if required
    if getattr(config, "P1_REQUIRED_XLK_POSITIVE", True):
        xlk_return = bot.etf_router.tape.xlk.return_bps() if bot.etf_router.tape.xlk else None
        if xlk_return is None or xlk_return <= 0:
            logger.info(f"P1: XLK return={xlk_return} bps, not positive, skipping")
            return False
    
    logger.warning(f"P1 FALLBACK TRIGGER: subtype={subtype}, XLK positive")
    return True


def execute_p1_entry(bot, current_time: dt_time) -> bool:
    """Execute P1 entry (TQQQ at 10:15).
    
    Returns True if entry succeeded.
    """
    if bot._check_daily_loss_kill_switch():
        logger.critical("P1 entry BLOCKED by daily-loss kill switch")
        return False
    
    vehicle = getattr(config, "P1_VEHICLE", "TQQQ")
    
    try:
        account = bot.position_mgr.get_account()
        if not account:
            logger.error("P1: cannot fetch account for sizing")
            return False
        
        equity = float(account.get("equity", 0) or 0)
        if equity <= 0:
            logger.error("P1: equity <= 0")
            return False
        
        # Size: 90% of equity for intraday ETF sleeve
        budget = equity * float(getattr(config, "INTRADAY_ETF_ALLOCATION_PCT", 0.90))
        
        # Get quote
        snapshots = bot.alpaca.get_snapshots([vehicle]) or {}
        snap = snapshots.get(vehicle, {}) or {}
        
        # Execution readiness check
        orderable, rejected = filter_execution_ready(
            [vehicle], snapshots,
            max_spread_pct=float(getattr(config, "ETF_ENTRY_MAX_SPREAD_PCT", 0.005)),
            require_quote=True,
            max_stale_seconds=float(getattr(config, "ETF_ENTRY_MAX_STALE_SECONDS", 10.0)),
        )
        if vehicle not in orderable:
            logger.warning(f"P1 entry rejected for {vehicle}: {rejected.get(vehicle, 'unknown')}")
            return False
        
        ask = snap.get("ask")
        last = snap.get("last_price") or snap.get("close")
        if not last:
            logger.error(f"P1: no last price for {vehicle}")
            return False
        
        sizing_price = float(ask) if ask else float(last)
        qty = int(budget / sizing_price)
        if qty <= 0:
            logger.warning(f"P1: qty <= 0 for {vehicle} budget=${budget:.2f} price={sizing_price:.4f}")
            return False
        
        # Per-branch SL/TP from config
        sl_tp_table = getattr(config, "ETF_SL_TP", {})
        sl_tp = sl_tp_table.get("P1_FALLBACK", {})
        sl_pct = sl_tp.get("sl")
        tp_pct = sl_tp.get("tp")

        # Submit order
        slippage_pct = float(getattr(config, "ETF_ENTRY_MAX_SLIPPAGE_PCT", 0.005))
        if ask:
            limit_price = float(ask) * (1.0 + slippage_pct)
            order_type = "limit"
            logger.warning(
                f"P1 BUY {vehicle} qty={qty} ask={ask} last={last} "
                f"limit={limit_price:.4f} SL={sl_pct} TP={tp_pct}"
            )
            order, error_type = bot.position_mgr.submit_bracket_buy_order(
                vehicle, qty,
                order_type="limit", limit_price=limit_price,
                stop_loss_pct=sl_pct, take_profit_pct=tp_pct,
                timeout=5,
            )
        else:
            order_type = "market"
            order, error_type = bot.position_mgr.submit_bracket_buy_order(
                vehicle, qty,
                order_type="market",
                stop_loss_pct=sl_pct, take_profit_pct=tp_pct,
                fill_price_hint=float(last),
                timeout=5,
            )
            limit_price = None

        if not order or not order.get("id"):
            logger.error(f"P1 buy failed for {vehicle}: {error_type}")
            return False

        # Wait for fill
        fill = bot.position_mgr.get_order_fill(order["id"], max_wait=10)
        if not fill or int(fill.get("filled_qty", 0)) <= 0:
            logger.warning(f"P1 buy not filled for {vehicle}; canceling")
            bot.position_mgr._cancel_order(order["id"])
            return False

        filled_qty = int(fill["filled_qty"])
        fill_price = float(fill.get("filled_avg_price", last))

        # P1 uses bracket order (SL/TP via submit_bracket_buy_order); hard time exit at 15:00 is fallback
        exit_t = _parse_hhmm(getattr(config, "P1_EXIT_TIME", "15:00"))

        # Record position
        bot.etf_position = {
            "symbol": vehicle,
            "qty": filled_qty,
            "entry_price": fill_price,
            "entry_time": datetime.now(_ET).isoformat(),
            "branch": "P1_FALLBACK",
            "planned_exit_time": exit_t.isoformat(),
            "order_id": order.get("id"),
            "entry_order_type": order_type,
            "entry_limit_price": limit_price if order_type == "limit" else None,
            "trailing_stop_order_id": None,  # No trailing stop for P1
            "trail_percent": None,
            "bracket_order_id": order.get("id") if (sl_pct or tp_pct) else None,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "mr_blocking": False,  # P1 does NOT block MR
        }
        bot.p1_active = True
        bot.intraday_etf_sleeve_filled = True  # Mark that we've used the intraday sleeve
        
        logger.warning(
            f"P1 POSITION OPENED: {vehicle} qty={filled_qty} fill={fill_price:.4f} exit={exit_t.isoformat()} MR_ALLOWED=True"
        )
        bot._save_state()
        return True
        
    except Exception:
        logger.error("P1 entry failed", exc_info=True)
        return False


def check_p1_exit(bot, current_time: dt_time) -> bool:
    """Check if P1 should be exited at its planned exit time (15:00).
    
    Returns True if exit time has been reached.
    """
    if not getattr(bot, "p1_active", False):
        return False
    
    if not bot.etf_position:
        return False
    
    exit_t = _parse_hhmm(getattr(config, "P1_EXIT_TIME", "15:00"))
    return current_time >= exit_t


def build_p1_summary(bot) -> Dict[str, Any]:
    """Build summary for EOD reporting."""
    return {
        "p1_enabled": getattr(config, "ENABLE_P1_FALLBACK", False),
        "p1_active": getattr(bot, "p1_active", False),
    }
