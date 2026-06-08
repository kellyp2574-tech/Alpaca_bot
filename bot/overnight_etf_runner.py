"""Overnight ETF strategies A/B/C — evaluated at 15:45 if no intraday trade.

Priority order (only ONE fires per day):
  A. VXX Mean Reversion  → BUY SVIX  (VXX >= +2.5% today)
  B. Overnight Quality   → BUY TQQQ  (SPY > +0.5% OR VXX < -2.0%, AND VXX < +2.0%)
  C. Gap Bounce          → BUY TQQQ  (QQQ < -0.5% today, QQQ 9:30-10:00 < 0%, VXX < 0%)

If any fires, `bot.overnight_etf_fired = True` which blocks single-stock MR for the day.
The position is stored in `bot.overnight_etf_position` and sold at market open (09:30) next day.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from bot import config
from bot.universe_builder import filter_execution_ready

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


def evaluate_overnight_etf_strategies(bot) -> None:
    """15:45: Evaluate strategies A/B/C in priority order.

    Sets bot.overnight_etf_fired = True and bot.overnight_etf_position if one fires.
    """
    if not getattr(config, "OVERNIGHT_ETF_ENABLED", True):
        logger.info("Overnight ETF sleeve disabled")
        return

    if getattr(bot, "overnight_etf_fired", False):
        logger.info("Overnight ETF already fired today — skipping re-evaluation")
        return

    logger.info("=" * 60)
    logger.info("OVERNIGHT ETF EVALUATION (15:45) — strategies A/B/C")
    logger.info("=" * 60)

    # Fetch snapshots once for all ETF symbols
    symbols = ["QQQ", "SPY", "VXX", "SVIX", "TQQQ"]
    try:
        snapshots = bot.alpaca.get_snapshots(symbols) or {}
    except Exception as e:
        logger.error(f"Cannot fetch snapshots for overnight ETF evaluation: {e}")
        return

    # README formula: day_return = (price_15:45 - price_9:30) / price_9:30
    # Use the 9:30 open prices from the tape when available; fall back to
    # prev_daily_close only if the tape wasn't seeded today (e.g. late start).
    router = getattr(bot, "etf_router", None)
    tape = router.tape if router and hasattr(router, "tape") else None

    def _tape_open(sym: str) -> Optional[float]:
        if tape is None:
            return None
        snap = getattr(tape, sym.lower(), None)
        return snap.open_930 if snap and snap.open_930 else None

    def day_ret(sym: str) -> Optional[float]:
        snap = snapshots.get(sym, {}) or {}
        last = snap.get("last_price")
        base = _tape_open(sym) or snap.get("prev_daily_close") or snap.get("prev_close")
        if last and base and float(base) > 0:
            return (float(last) - float(base)) / float(base) * 100.0
        return None

    qqq_ret = day_ret("QQQ")
    spy_ret = day_ret("SPY")
    vxx_ret = day_ret("VXX")

    def _fmt(v) -> str:
        return f"{v:.2f}%" if v is not None else "N/A"
    logger.info(
        f"Day returns (from 9:30) — QQQ={_fmt(qqq_ret)} SPY={_fmt(spy_ret)} VXX={_fmt(vxx_ret)}"
    )

    # QQQ 9:30-10:00 return from the tape (Strategy C condition)
    qqq_0930_ret: Optional[float] = None
    if tape and tape.qqq.is_valid():
        qqq_0930_ret = tape.qqq.return_pct()

    # ── Strategy A: VXX Mean Reversion ───────────────────────────────────────
    vxx_trigger = float(getattr(config, "OVERNIGHT_VXX_MR_TRIGGER_PCT", 2.5))
    vxx_vehicle = getattr(config, "OVERNIGHT_VXX_MR_VEHICLE", "SVIX")

    if vxx_ret is not None and vxx_ret >= vxx_trigger:
        logger.info(
            f"Strategy A (VXX MR): VXX={vxx_ret:.2f}% >= {vxx_trigger:.1f}% trigger "
            f"→ BUY {vxx_vehicle}"
        )
        _execute_overnight_etf_entry(bot, vxx_vehicle, "OVERNIGHT_VXX_MR", snapshots)
        return

    # ── Strategy B: Overnight Quality ────────────────────────────────────────
    spy_min     = float(getattr(config, "OVERNIGHT_QUALITY_SPY_MIN_PCT", 0.5))
    vxx_collapse= float(getattr(config, "OVERNIGHT_QUALITY_VXX_COLLAPSE_PCT", -2.0))
    vxx_excl    = float(getattr(config, "OVERNIGHT_QUALITY_VXX_EXCLUSION_PCT", 2.0))
    quality_vehicle = getattr(config, "OVERNIGHT_QUALITY_VEHICLE", "TQQQ")

    quality_signal = (
        (spy_ret is not None and spy_ret > spy_min)
        or (vxx_ret is not None and vxx_ret < vxx_collapse)
    )
    vxx_safe = vxx_ret is None or vxx_ret < vxx_excl

    if quality_signal and vxx_safe:
        logger.info(
            f"Strategy B (Quality): SPY={spy_ret:.2f}% VXX={vxx_ret:.2f}% "
            f"→ BUY {quality_vehicle}"
        )
        _execute_overnight_etf_entry(bot, quality_vehicle, "OVERNIGHT_QUALITY", snapshots)
        return

    # ── Strategy C: Gap Bounce ────────────────────────────────────────────────
    qqq_max_pct = float(getattr(config, "OVERNIGHT_GAP_BOUNCE_QQQ_MAX_PCT", -0.5))
    gap_vehicle = getattr(config, "OVERNIGHT_GAP_BOUNCE_VEHICLE", "TQQQ")

    gap_signal = (
        qqq_ret is not None and qqq_ret < qqq_max_pct
        and vxx_ret is not None and vxx_ret < 0.0
        and (qqq_0930_ret is None or qqq_0930_ret < 0.0)  # 9:30-10:00 also negative
    )

    if gap_signal:
        logger.info(
            f"Strategy C (Gap Bounce): QQQ={qqq_ret:.2f}% VXX={vxx_ret:.2f}% "
            f"QQQ_0930_ret={qqq_0930_ret} → BUY {gap_vehicle}"
        )
        _execute_overnight_etf_entry(bot, gap_vehicle, "OVERNIGHT_GAP_BOUNCE", snapshots)
        return

    logger.info("Overnight ETF: no strategy fired — single-stock MR may proceed")


def _execute_overnight_etf_entry(
    bot,
    symbol: str,
    strategy_name: str,
    snapshots: Dict[str, Any],
) -> None:
    """Submit overnight ETF buy order using 90% of equity."""
    if bot._check_daily_loss_kill_switch():
        logger.critical(
            f"Overnight ETF entry BLOCKED by kill switch — {bot.kill_switch_reason}; "
            f"would have bought {symbol} ({strategy_name})"
        )
        return

    try:
        account = bot.position_mgr.get_account()
        if not account:
            logger.error("Cannot fetch account for overnight ETF sizing")
            return

        equity = float(account.get("equity", 0))
        alloc_pct = float(getattr(config, "INTRADAY_ETF_ALLOCATION_PCT", 0.90))
        budget = equity * alloc_pct

        snap = snapshots.get(symbol, {}) or {}
        orderable, rejected = filter_execution_ready(
            [symbol], snapshots,
            max_spread_pct=float(getattr(config, "ETF_ENTRY_MAX_SPREAD_PCT", 0.005)),
            require_quote=True,
            max_stale_seconds=float(getattr(config, "ETF_ENTRY_MAX_STALE_SECONDS", 60.0)),
        )
        if symbol not in orderable:
            logger.warning(f"Overnight ETF entry REJECTED for {symbol}: {rejected.get(symbol)}")
            return

        ask = snap.get("ask")
        last_price = snap.get("last_price")
        if not last_price:
            logger.error(f"Overnight ETF {symbol}: no last_price")
            return

        sizing_price = float(ask) if ask else float(last_price)
        qty = int(budget / sizing_price)

        if qty <= 0:
            logger.warning(f"Overnight ETF qty=0 for {symbol}: budget={budget:.2f} price={sizing_price:.2f}")
            return

        slippage = float(getattr(config, "ETF_ENTRY_MAX_SLIPPAGE_PCT", 0.005))
        if ask:
            order, err = bot.position_mgr.submit_bracket_buy_order(
                symbol, qty,
                order_type="limit",
                limit_price=float(ask) * (1.0 + slippage),
            )
        else:
            order, err = bot.position_mgr.submit_bracket_buy_order(
                symbol, qty,
                order_type="market",
                fill_price_hint=float(last_price),
            )

        if order and order.get("id"):
            fill = bot.position_mgr.get_order_fill(order["id"], max_wait=10)
            if fill and int(fill.get("filled_qty", 0)) > 0:
                filled_qty  = int(fill["filled_qty"])
                fill_price  = float(fill.get("filled_avg_price", last_price))
                exit_time   = getattr(config, "OVERNIGHT_ETF_EXIT_TIME", "09:30")
                bot.overnight_etf_position = {
                    "symbol":       symbol,
                    "qty":          filled_qty,
                    "entry_price":  fill_price,
                    "entry_time":   datetime.now(_ET).isoformat(),
                    "strategy":     strategy_name,
                    "planned_exit": exit_time,
                    "order_id":     order.get("id"),
                }
                bot.overnight_etf_fired = True
                logger.info(
                    f"Overnight ETF filled: {symbol} {filled_qty} @ ${fill_price:.2f} "
                    f"({strategy_name}) — single-stock MR BLOCKED"
                )
                # Update _exec_stats so run_health shows correct deployment on overnight-ETF days.
                deployed = filled_qty * fill_price
                existing = getattr(bot, "_exec_stats", {}) or {}
                bot._exec_stats = dict(existing)
                bot._exec_stats["equity"] = equity
                bot._exec_stats["selected"] = existing.get("selected", 0) + 1
                bot._exec_stats["orderable"] = existing.get("orderable", 0) + 1
                bot._exec_stats["orders_submitted"] = existing.get("orders_submitted", 0) + 1
                bot._exec_stats["overnight_etf_filled"] = {
                    "symbol": symbol,
                    "qty": filled_qty,
                    "price": round(fill_price, 4),
                    "strategy": strategy_name,
                    "deployed": round(deployed, 2),
                }
                bot._exec_stats["total_deployed"] = (
                    float(existing.get("total_deployed", 0.0)) + deployed
                )
                bot._exec_stats["entries_filled"] = (
                    int(existing.get("entries_filled", 0)) + 1
                )
                bot._save_state()
            else:
                logger.warning(f"Overnight ETF order not filled for {symbol}")
                try:
                    bot.position_mgr._cancel_order(order["id"])
                except Exception:
                    pass
        else:
            logger.error(f"Overnight ETF order submission failed for {symbol}: {err}")

    except Exception as e:
        logger.error(f"Error executing overnight ETF entry: {e}", exc_info=True)
