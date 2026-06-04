"""V2 Fallback Module — Priority 2 intraday sleeve (Router > V2 > P1)

V2 evaluates when the primary ETF router returns NO_TRADE.
- V2 Long: QQQ breaks above 09:30-10:00 morning high, Set C subtype, 10:10-10:30 window
- V2 Short: QQQ/SPY/IWM all break below 09:30-10:00 lows, Set C subtype, 10:15-10:30 window
- Entry uses 90% ETF bucket (same as router)
- V2 Long: trailing stop after entry (V2_TRAIL_PCT=1.5%)
- V2 Short: bracket order with SL=-2% / TP=+3% (per ETF_SL_TP["V2_SHORT"])
- 11:30 hybrid checkpoint: exit if red or trend fails, hold if green
- Hard exit at 15:30 if still holding

Every function takes the orchestrator instance (`bot`) as first argument
and mutates state directly. State ownership stays with the orchestrator.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from bot import config
from bot.universe_builder import filter_execution_ready

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


def _parse_hhmm(value: str) -> dt_time:
    """Parse 'HH:MM' config string into time object."""
    parts = str(value).split(":")
    return dt_time(int(parts[0]), int(parts[1]))


def _get_qqq_morning_range(bot) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (high, low, open_930) for QQQ from the recorded tape."""
    q = bot.etf_router.tape.qqq
    if q.open_930 and q.high and q.low:
        return q.high, q.low, q.open_930
    return None, None, None


def _get_spy_iwm_morning_lows(bot) -> Tuple[Optional[float], Optional[float]]:
    """Return (spy_low, iwm_low) from the recorded tape."""
    spy = bot.etf_router.tape.spy
    iwm = bot.etf_router.tape.iwm
    spy_low = spy.low if spy and spy.low else None
    iwm_low = iwm.low if iwm and iwm.low else None
    return spy_low, iwm_low


def _is_v2_subtype_allowed(subtype: Optional[str]) -> bool:
    """Check if the no-trade subtype allows V2 fallback."""
    allowed = set(getattr(config, "V2_ALLOWED_SUBTYPES", []))
    if not allowed:
        return True  # If not configured, allow all
    return subtype in allowed


def evaluate_v2_long(bot, current_time: dt_time, snapshots: Optional[Dict[str, dict]] = None) -> bool:
    """Evaluate V2 long entry conditions.
    
    Returns True if V2 long should fire (QQQ breaks above morning high).
    Called during 10:10-10:30 window when router returned no-trade.
    
    Args:
        bot: Bot instance
        current_time: Current time
        snapshots: Optional pre-fetched snapshot dict (for throttled evaluation)
    """
    if not getattr(config, "ENABLE_V2_FALLBACK", False):
        return False
    
    # Check time window
    start_t = _parse_hhmm(getattr(config, "V2_LONG_ENTRY_WINDOW_START", "10:10"))
    end_t = _parse_hhmm(getattr(config, "V2_LONG_ENTRY_WINDOW_END", "10:30"))
    if not (start_t <= current_time <= end_t):
        return False
    
    # Check subtype eligibility
    subtype = getattr(bot, "router_no_trade_subtype", None)
    if not _is_v2_subtype_allowed(subtype):
        logger.info(f"V2 long not allowed: subtype={subtype} not in allowed set")
        return False
    
    # Get QQQ morning range
    q_high, q_low, q_open = _get_qqq_morning_range(bot)
    if q_high is None or q_low is None or q_open is None:
        logger.warning("V2 long: missing QQQ morning range data")
        return False
    
    # Check range threshold
    threshold = float(getattr(config, "V2_QQQ_RANGE_BREAKOUT_THRESHOLD", 0.0045))
    q_range_pct = (q_high - q_low) / q_open if q_open > 0 else 0
    if q_range_pct < threshold:
        logger.info(f"V2 long: QQQ range {q_range_pct:.3%} < threshold {threshold:.3%}")
        return False
    
    # Get current QQQ price (use provided snapshots or fetch)
    try:
        snaps = snapshots if snapshots is not None else bot.alpaca.get_snapshots(["QQQ"]) or {}
        q_snap = snaps.get("QQQ", {}) or {}
        q_price = q_snap.get("last_price") or q_snap.get("close")
        if not q_price:
            logger.warning("V2 long: no QQQ last price available")
            return False
        q_price = float(q_price)
    except Exception:
        logger.warning("V2 long: failed to get QQQ snapshot", exc_info=True)
        return False
    
    # Check breakout above morning high
    if q_price > q_high:
        logger.warning(f"V2 LONG TRIGGER: QQQ {q_price:.4f} > morning high {q_high:.4f}")
        bot.v2_trigger_price = q_price
        bot.v2_trigger_high = q_high
        bot.v2_trigger_low = q_low
        bot.v2_trigger_range_pct = q_range_pct
        return True
    
    return False


def evaluate_v2_short(bot, current_time: dt_time, snapshots: Optional[Dict[str, dict]] = None) -> bool:
    """Evaluate V2 short entry conditions.
    
    Returns True if V2 short should fire (QQQ/SPY/IWM all break below morning lows).
    Called during 10:15-10:30 window when router returned no-trade and no V2 long.
    
    Args:
        bot: Bot instance
        current_time: Current time
        snapshots: Optional pre-fetched snapshot dict (for throttled evaluation)
    """
    if not getattr(config, "ENABLE_V2_FALLBACK", False):
        return False
    
    # Check time window
    start_t = _parse_hhmm(getattr(config, "V2_SHORT_ENTRY_WINDOW_START", "10:15"))
    end_t = _parse_hhmm(getattr(config, "V2_SHORT_ENTRY_WINDOW_END", "10:30"))
    if not (start_t <= current_time <= end_t):
        return False
    
    # Check subtype eligibility
    subtype = getattr(bot, "router_no_trade_subtype", None)
    if not _is_v2_subtype_allowed(subtype):
        logger.info(f"V2 short not allowed: subtype={subtype} not in allowed set")
        return False
    
    # Get morning ranges
    q_high, q_low, q_open = _get_qqq_morning_range(bot)
    spy_low, iwm_low = _get_spy_iwm_morning_lows(bot)
    
    if None in (q_high, q_low, q_open, spy_low, iwm_low):
        logger.warning("V2 short: missing ETF morning range data")
        return False
    
    # Check range threshold for QQQ
    threshold = float(getattr(config, "V2_QQQ_RANGE_BREAKOUT_THRESHOLD", 0.0045))
    q_range_pct = (q_high - q_low) / q_open if q_open > 0 else 0
    if q_range_pct < threshold:
        logger.info(f"V2 short: QQQ range {q_range_pct:.3%} < threshold {threshold:.3%}")
        return False
    
    # Get current prices (use provided snapshots or fetch)
    try:
        snaps = snapshots if snapshots is not None else bot.alpaca.get_snapshots(["QQQ", "SPY", "IWM"]) or {}
        q_price = snaps.get("QQQ", {}).get("last_price") or snaps.get("QQQ", {}).get("close")
        spy_price = snaps.get("SPY", {}).get("last_price") or snaps.get("SPY", {}).get("close")
        iwm_price = snaps.get("IWM", {}).get("last_price") or snaps.get("IWM", {}).get("close")
        
        if None in (q_price, spy_price, iwm_price):
            logger.warning("V2 short: missing ETF price data")
            return False
        
        q_price = float(q_price)
        spy_price = float(spy_price)
        iwm_price = float(iwm_price)
    except Exception:
        logger.warning("V2 short: failed to get ETF snapshots", exc_info=True)
        return False
    
    # Check breakdown below all three morning lows
    if q_price < q_low and spy_price < spy_low and iwm_price < iwm_low:
        logger.warning(
            f"V2 SHORT TRIGGER: QQQ {q_price:.4f} < {q_low:.4f}, "
            f"SPY {spy_price:.4f} < {spy_low:.4f}, IWM {iwm_price:.4f} < {iwm_low:.4f}"
        )
        bot.v2_trigger_price = q_price
        bot.v2_trigger_high = q_high
        bot.v2_trigger_low = q_low
        bot.v2_trigger_range_pct = q_range_pct
        return True
    
    return False


def execute_v2_entry(bot, direction: str, current_time: dt_time) -> bool:
    """Execute V2 entry (long or short).
    
    Args:
        direction: "long" or "short"
        current_time: Current time for logging
    
    Returns True if entry succeeded.
    """
    if bot._check_daily_loss_kill_switch():
        logger.critical(f"V2 {direction} BLOCKED by daily-loss kill switch")
        return False
    
    vehicle = (
        getattr(config, "V2_LONG_VEHICLE", "TQQQ") 
        if direction == "long" 
        else getattr(config, "V2_SHORT_VEHICLE", "SPXU")
    )
    
    try:
        account = bot.position_mgr.get_account()
        if not account:
            logger.error(f"V2 {direction}: cannot fetch account for sizing")
            return False
        
        equity = float(account.get("equity", 0) or 0)
        if equity <= 0:
            logger.error(f"V2 {direction}: equity <= 0")
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
            logger.warning(f"V2 {direction} entry rejected for {vehicle}: {rejected.get(vehicle, 'unknown')}")
            return False
        
        ask = snap.get("ask")
        last = snap.get("last_price") or snap.get("close")
        if not last:
            logger.error(f"V2 {direction}: no last price for {vehicle}")
            return False
        
        sizing_price = float(ask) if ask else float(last)
        qty = int(budget / sizing_price)
        if qty <= 0:
            logger.warning(f"V2 {direction}: qty <= 0 for {vehicle} budget=${budget:.2f} price={sizing_price:.4f}")
            return False
        
        # Per-branch SL/TP: V2 Long leaves trailing stop; V2 Short uses bracket
        sl_tp_table = getattr(config, "ETF_SL_TP", {})
        branch_key = f"V2_{direction.upper()}"  # "V2_LONG" or "V2_SHORT"
        sl_tp = sl_tp_table.get(branch_key, {})
        sl_pct = sl_tp.get("sl")
        tp_pct = sl_tp.get("tp")

        # Submit order
        slippage_pct = float(getattr(config, "ETF_ENTRY_MAX_SLIPPAGE_PCT", 0.005))
        if ask:
            limit_price = float(ask) * (1.0 + slippage_pct)
            order_type = "limit"
            logger.warning(
                f"V2 {direction.upper()} BUY {vehicle} qty={qty} ask={ask} last={last} "
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
            logger.error(f"V2 {direction} buy failed for {vehicle}: {error_type}")
            return False

        # Wait for fill
        fill = bot.position_mgr.get_order_fill(order["id"], max_wait=10)
        if not fill or int(fill.get("filled_qty", 0)) <= 0:
            logger.warning(f"V2 {direction} buy not filled for {vehicle}; canceling")
            bot.position_mgr._cancel_order(order["id"])
            return False

        filled_qty = int(fill["filled_qty"])
        fill_price = float(fill.get("filled_avg_price", last))

        # V2 Long: trailing stop (unchanged per backtest table)
        # V2 Short: bracket handles exits, no trailing stop
        trailing_id = None
        trail_pct = None
        if direction == "long":
            trail_pct = float(getattr(config, "V2_TRAIL_PCT", 1.50))
            trailing_order = bot.position_mgr.submit_trailing_stop_sell_order(vehicle, filled_qty, trail_pct)
            trailing_id = trailing_order.get("id") if trailing_order else None
            if not trailing_id:
                logger.error(f"V2 long: trailing stop submit failed for {vehicle}; will use hard exit")

        # Record position
        bot.etf_position = {
            "symbol": vehicle,
            "qty": filled_qty,
            "entry_price": fill_price,
            "entry_time": datetime.now(_ET).isoformat(),
            "branch": f"V2_{direction.upper()}",
            "planned_exit_time": _parse_hhmm(getattr(config, "V2_FINAL_EXIT_TIME", "15:30")).isoformat(),
            "order_id": order.get("id"),
            "entry_order_type": order_type,
            "entry_limit_price": limit_price if order_type == "limit" else None,
            "trailing_stop_order_id": trailing_id,
            "trail_percent": trail_pct,
            "bracket_order_id": order.get("id") if (sl_pct or tp_pct) else None,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "mr_blocking": False,  # V2 does NOT block MR
            "trigger_price": getattr(bot, "v2_trigger_price", None),
            "trigger_high": getattr(bot, "v2_trigger_high", None),
            "trigger_low": getattr(bot, "v2_trigger_low", None),
            "trigger_range_pct": getattr(bot, "v2_trigger_range_pct", None),
            "v2_direction": direction,
        }
        bot.v2_active = True
        bot.v2_direction = direction
        bot.intraday_etf_sleeve_filled = True  # Mark that we've used the intraday sleeve

        logger.warning(
            f"V2 {direction.upper()} POSITION OPENED: {vehicle} qty={filled_qty} "
            f"fill={fill_price:.4f} trail={trail_pct} SL={sl_pct} TP={tp_pct} exit=15:30 MR_ALLOWED=True"
        )
        bot._save_state()
        return True
        
    except Exception:
        logger.error(f"V2 {direction} entry failed", exc_info=True)
        return False


def evaluate_v2_hybrid_checkpoint(bot, current_time: dt_time) -> bool:
    """Evaluate V2 at 11:30 hybrid checkpoint.
    
    For V2 long: exit if red at 11:30 or trend confirmation fails
    For V2 short: exit if red at 11:30 or downside confirmation fails
    
    Returns True if position should be exited.
    """
    if not getattr(bot, "v2_active", False):
        return False
    
    checkpoint_t = _parse_hhmm(getattr(config, "V2_HYBRID_CHECKPOINT_TIME", "11:30"))
    if current_time < checkpoint_t:
        return False
    
    if not bot.etf_position:
        return False
    
    symbol = bot.etf_position.get("symbol")
    direction = bot.etf_position.get("v2_direction", "long")
    entry_price = bot.etf_position.get("entry_price", 0)
    
    if not symbol or not entry_price:
        return False
    
    try:
        # Get current price
        snaps = bot.alpaca.get_snapshots([symbol]) or {}
        snap = snaps.get(symbol, {}) or {}
        current_price = snap.get("last_price") or snap.get("close")
        if not current_price:
            logger.warning(f"V2 checkpoint: no current price for {symbol}")
            return False
        current_price = float(current_price)
        
        # Check if position is "red" (losing money)
        is_red = current_price < entry_price
        
        if is_red:
            logger.warning(f"V2 {direction} EXIT at 11:30: red position {current_price:.4f} < entry {entry_price:.4f}")
            return True
        
        # Trend confirmation check (simplified)
        # For long: check if QQQ is still above morning high
        # For short: check if QQQ is still below morning low
        qqq_snaps = bot.alpaca.get_snapshots(["QQQ"]) or {}
        qqq_price = qqq_snaps.get("QQQ", {}).get("last_price")
        
        if qqq_price:
            qqq_price = float(qqq_price)
            q_high = bot.etf_position.get("trigger_high")
            q_low = bot.etf_position.get("trigger_low")
            
            if direction == "long" and q_high and qqq_price < q_high * 0.995:  # Slight buffer
                logger.warning(f"V2 long EXIT at 11:30: QQQ {qqq_price:.4f} below confirmation level")
                return True
            elif direction == "short" and q_low and qqq_price > q_low * 1.005:
                logger.warning(f"V2 short EXIT at 11:30: QQQ {qqq_price:.4f} above confirmation level")
                return True
        
        logger.info(f"V2 {direction} HOLD at 11:30: position green and trend confirmed")
        return False
        
    except Exception:
        logger.error("V2 checkpoint evaluation failed", exc_info=True)
        return False  # Conservative: don't exit on error


def check_v2_final_exit(bot, current_time: dt_time) -> bool:
    """Check if V2 should be hard-exited at 15:30 deadline."""
    if not getattr(bot, "v2_active", False):
        return False
    
    final_exit_t = _parse_hhmm(getattr(config, "V2_FINAL_EXIT_TIME", "15:30"))
    return current_time >= final_exit_t


def build_v2_summary(bot) -> Dict[str, Any]:
    """Build summary for EOD reporting."""
    return {
        "v2_enabled": getattr(config, "ENABLE_V2_FALLBACK", False),
        "v2_active": getattr(bot, "v2_active", False),
        "v2_direction": getattr(bot, "v2_direction", None),
        "v2_trigger_price": getattr(bot, "v2_trigger_price", None),
        "v2_trigger_high": getattr(bot, "v2_trigger_high", None),
        "v2_trigger_low": getattr(bot, "v2_trigger_low", None),
        "v2_trigger_range_pct": getattr(bot, "v2_trigger_range_pct", None),
    }
