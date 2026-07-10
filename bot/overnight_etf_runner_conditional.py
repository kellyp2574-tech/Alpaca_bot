"""Conditional TQQQ Overnight Strategy.

Sole active overnight strategy: TQQQ at 75% of equity when conditions are met.
Single-stock MR overnight is disabled (MR_OVERNIGHT_ENABLED=False).

Conditions for TQQQ entry:
1. TQQQ expected return > minimum threshold (>0.2%), OR
2. TQQQ shows strong expected return (>1.5%)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from bot import config
from bot.universe_builder import filter_execution_ready

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


def evaluate_overnight_etf_strategies(bot) -> None:
    """15:45: Evaluate and execute TQQQ overnight strategy at 75% sizing."""
    if not getattr(config, "OVERNIGHT_ETF_ENABLED", True):
        logger.info("Overnight ETF sleeve disabled")
        return

    logger.info("=" * 60)
    logger.info("TQQQ OVERNIGHT EVALUATION (15:45)")
    logger.info("=" * 60)

    # Evaluate TQQQ signal
    tqqq_signal, tqqq_expected_return = _evaluate_tqqq_signal(bot)
    tqqq_allocation = _calculate_tqqq_allocation(tqqq_signal, tqqq_expected_return)

    logger.info(f"TQQQ signal: {'POSITIVE' if tqqq_signal else 'NEGATIVE/NEUTRAL'} (expected: {tqqq_expected_return:+.3%})")
    logger.info(f"TQQQ allocation: {tqqq_allocation:.1%}")

    if tqqq_allocation > 0:
        _execute_conditional_tqqq_entry(bot, tqqq_allocation)
    else:
        logger.info("TQQQ conditions not met — no overnight position")
        bot.overnight_etf_fired = False
        bot.overnight_etf_position = None

    bot.overnight_etf_decision_made = True
    bot._save_state()


def _evaluate_tqqq_signal(bot) -> Tuple[bool, float]:
    """Evaluate TQQQ overnight signal and expected return.
    
    Returns:
        Tuple[signal_positive, expected_return]
    """
    try:
        # Fetch market data
        symbols = ["QQQ", "SPY", "VXX", "TQQQ"]
        snapshots = bot.alpaca.get_snapshots(symbols) or {}
        
        if not snapshots.get("QQQ") or not snapshots.get("VXX"):
            logger.warning("Missing QQQ or VXX data for TQQQ signal evaluation")
            return False, 0.0
        
        # Calculate day returns
        def day_ret(sym: str) -> Optional[float]:
            snap = snapshots.get(sym, {}) or {}
            last = snap.get("last_price")
            base = snap.get("prev_daily_close") or snap.get("prev_close")
            if last and base and float(base) > 0:
                return (float(last) - float(base)) / float(base)
            return None
        
        qqq_ret = day_ret("QQQ")
        spy_ret = day_ret("SPY")
        vxx_ret = day_ret("VXX")
        
        if qqq_ret is None or vxx_ret is None:
            logger.warning("Cannot calculate QQQ or VXX returns")
            return False, 0.0
        
        # Get VIX level - prefer actual VIX data, fallback to None (skip VIX factor)
        vix_level = None
        # Try to get cached VIX from morning MR if available
        if hasattr(bot, 'intraday_mr_vix_open') and bot.intraday_mr_vix_open:
            vix_level = float(bot.intraday_mr_vix_open)
            logger.info(f"Using cached VIX from morning MR: {vix_level:.2f}")
        # Note: VXX price cannot reliably estimate spot VIX due to futures term structure,
        # path dependency, fees, and reverse splits. Skip VIX factor if no actual VIX available.
        
        # TQQQ signal logic based on analysis findings
        signal_positive = False
        expected_return = 0.0
        
        # Factor 1: VIX regime (low VIX = trending = good for TQQQ)
        vix_low_threshold = float(getattr(config, "TQQQ_VIX_LOW_THRESHOLD", 15.0))
        vix_high_threshold = float(getattr(config, "TQQQ_VIX_HIGH_THRESHOLD", 25.0))
        
        if vix_level:
            if vix_level < vix_low_threshold:
                expected_return += 0.005  # Low VIX bonus
                signal_positive = True
            elif vix_level > vix_high_threshold:
                expected_return -= 0.003  # High VIX penalty
        
        # Factor 2: QQQ performance (moderate positive = good, extreme = bad)
        if qqq_ret > 0.01:  # Strong up day
            expected_return += 0.002
        elif qqq_ret < -0.015:  # Strong down day (mean reversion opportunity)
            expected_return += 0.008
            signal_positive = True
        elif -0.005 <= qqq_ret <= 0.01:  # Moderate positive
            expected_return += 0.003
            signal_positive = True
        
        # Factor 3: VXX performance (VXX down = good for TQQQ)
        if vxx_ret < -0.02:  # VXX collapse
            expected_return += 0.006
            signal_positive = True
        elif vxx_ret > 0.02:  # VXX spike (risk off)
            expected_return -= 0.004
        
        # Factor 4: SPY trend (positive trend = good)
        if spy_ret and spy_ret > 0.005:
            expected_return += 0.002
            signal_positive = True
        
        # Cap expected return at reasonable bounds
        expected_return = max(-0.02, min(0.03, expected_return))
        
        # Final signal determination
        signal_positive = signal_positive and expected_return > 0.002  # Minimum positive expectation
        
        vix_str = f"{vix_level:.1f}" if vix_level else "N/A"
        logger.info(f"TQQQ signal analysis: QQQ={qqq_ret:+.3%}, VXX={vxx_ret:+.3%}, VIX={vix_str}, expected={expected_return:+.3%}")
        
        return signal_positive, expected_return
        
    except Exception as e:
        logger.error(f"Error evaluating TQQQ signal: {e}")
        return False, 0.0


def _calculate_tqqq_allocation(tqqq_positive: bool, tqqq_expected: float) -> float:
    """Return TQQQ allocation (75%) when signal is positive, else 0."""
    allocation = float(getattr(config, "TQQQ_CONDITIONAL_ALLOCATION_PCT", 0.75))
    if tqqq_positive:
        return allocation
    return 0.0


def _execute_conditional_tqqq_entry(bot, allocation_pct: float) -> None:
    """Execute TQQQ entry with specified allocation percentage."""
    if bot._check_daily_loss_kill_switch():
        logger.critical(
            f"Conditional TQQQ entry BLOCKED by kill switch — {bot.kill_switch_reason}"
        )
        return

    try:
        account = bot.position_mgr.get_account()
        if not account:
            logger.error("Cannot fetch account for TQQQ sizing")
            return

        equity = float(account.get("equity", 0))
        budget = equity * allocation_pct
        
        logger.info(f"TQQQ conditional entry: {allocation_pct:.1%} of equity = ${budget:,.2f}")

        # Get TQQQ snapshot
        snapshots = bot.alpaca.get_snapshots(["TQQQ"]) or {}
        tqqq_snap = snapshots.get("TQQQ", {})
        
        # Check execution readiness
        orderable, rejected = filter_execution_ready(
            ["TQQQ"], snapshots,
            max_spread_pct=float(getattr(config, "ETF_ENTRY_MAX_SPREAD_PCT", 0.005)),
            require_quote=True,
            max_stale_seconds=float(getattr(config, "ETF_ENTRY_MAX_STALE_SECONDS", 60.0)),
        )
        
        if "TQQQ" not in orderable:
            logger.warning(f"TQQQ entry REJECTED: {rejected.get('TQQQ')}")
            return

        # Get price
        ask = tqqq_snap.get("ask")
        last_price = tqqq_snap.get("last_price")
        price = ask if ask and ask > 0 else last_price
        
        if not price:
            logger.error("TQQQ: no valid price")
            return

        # Calculate shares with buying power buffer (no 100-share lot requirement)
        bp_buffer = float(getattr(config, "ENTRY_BP_BUFFER_PCT", 0.98))
        shares = int(budget * bp_buffer / float(price))
        if shares <= 0:
            logger.warning(f"TQQQ: calculated shares={shares}, budget=${budget:.2f}, price=${price:.2f}")
            return

        # Submit order with marketable limit (bounds slippage like intraday router)
        slippage_pct = float(getattr(config, "ETF_ENTRY_MAX_SLIPPAGE_PCT", 0.005))
        limit_price = float(ask) * (1.0 + slippage_pct) if ask else float(price) * (1.0 + slippage_pct)

        submitted_order, error_type = bot.position_mgr.submit_buy_order(
            symbol="TQQQ",
            qty=shares,
            order_type="limit",
            limit_price=limit_price,
        )

        if not submitted_order or not submitted_order.get("id"):
            logger.error(f"TQQQ conditional entry FAILED: {error_type}")
            return
        
        # Wait for fill confirmation (same pattern as intraday router)
        fill = bot.position_mgr.get_order_fill(submitted_order["id"], max_wait=10)
        if not fill or int(fill.get("filled_qty", 0)) <= 0:
            logger.warning("TQQQ conditional entry not filled - canceling and leaving flat")
            bot.position_mgr._cancel_order(submitted_order["id"])
            return
        
        filled_qty = int(fill["filled_qty"])
        fill_price = float(fill.get("filled_avg_price", price))
        actual_allocation = (filled_qty * fill_price) / equity if equity > 0 else 0
        
        logger.info(
            f"TQQQ conditional entry FILLED: {filled_qty} shares @ ${fill_price:.2f} "
            f"(actual allocation={actual_allocation:.1%}, budget=${budget:,.2f})"
        )
        
        # Store confirmed position info for morning exit
        bot.overnight_etf_fired = True
        bot.overnight_etf_position = {
            "symbol": "TQQQ",
            "qty": filled_qty,
            "entry_price": fill_price,
            "entry_time": datetime.now(_ET).isoformat(),
            "strategy": "CONDITIONAL_TQQQ",
            "allocation_pct": actual_allocation,  # Store actual, not intended
            "order_id": submitted_order["id"],
        }
            
    except Exception as e:
        logger.error(f"Error executing TQQQ conditional entry: {e}")


def get_overnight_allocation_summary(bot) -> Dict[str, float]:
    """Get summary of current overnight allocations (TQQQ-only design).

    Returns:
        Dict with allocation percentages for TQQQ and cash.
    """
    tqqq_allocation = 0.0

    if hasattr(bot, "overnight_etf_position") and bot.overnight_etf_position:
        if bot.overnight_etf_position.get("symbol") == "TQQQ":
            tqqq_allocation = bot.overnight_etf_position.get("allocation_pct", 0.0)

    return {
        "tqqq": tqqq_allocation,
        "cash": max(0.0, 1.0 - tqqq_allocation),
        "total": tqqq_allocation,
    }
