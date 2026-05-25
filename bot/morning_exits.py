"""Morning exit pipeline — extracted from integrated_main.py.

Owns the 09:25 → ~09:45 exit sequence: sleeve exits, single-position exits,
open-exit plan construction, batched market-sell submission, broker-native
rescue pass, red-open trailing-stop decision, and the final failsafe flatten.

Every function takes the orchestrator instance (`bot`) as its first argument
and mutates its state directly. State ownership stays with
`CombinedOvernightReboundBot`; this module only houses the logic.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from bot import config

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


# ────────────────────────────────────────────────────────────────────
# Sleeve / single-position exits
# ────────────────────────────────────────────────────────────────────

def exit_sleeve_positions(bot, sleeve: str, reason: str) -> None:
    """Exit all positions of a specific sleeve at market.

    Batch behavior: submit every sell order first, then monitor fills.
    This prevents one slow/partial fill from delaying the next symbol's
    sell order by minutes at the open.
    """
    # For GDP exit, also include UNKNOWN sleeve positions (safer early exit)
    if sleeve == "GDP":
        positions = [
            symbol for symbol, pos in bot.position_mgr.positions.items()
            if getattr(pos, "sleeve", "MR") == sleeve
            or getattr(pos, "sleeve", "MR") == "UNKNOWN"
        ]
        unknown_positions = [
            symbol for symbol, pos in bot.position_mgr.positions.items()
            if getattr(pos, "sleeve", "MR") == "UNKNOWN"
        ]
        if unknown_positions:
            logger.warning(
                f"EXIT {sleeve}: including {len(unknown_positions)} UNKNOWN sleeve positions: {unknown_positions}"
            )
    else:
        positions = [
            symbol for symbol, pos in bot.position_mgr.positions.items()
            if getattr(pos, "sleeve", "MR") == sleeve
        ]

    if not positions:
        logger.info(f"EXIT {sleeve}: no positions to sell")
        return

    logger.info(f"EXIT {sleeve}: market selling {len(positions)} positions: {positions}")

    # Pass 1: submit all sell orders
    submitted_orders: List[Tuple[str, str, int]] = []

    for symbol in positions:
        position = bot.position_mgr.positions.get(symbol)
        if not position:
            continue

        # Verify broker actually holds this position
        broker_pos = bot.position_mgr.get_broker_position(symbol)
        if broker_pos is None:
            logger.warning(f"EXIT DEFER {symbol}: broker API error — skipping for retry")
            continue
        if broker_pos is bot.position_mgr.BROKER_NOT_FOUND:
            logger.warning(f"EXIT SKIP {symbol}: broker confirmed no position — removing local")
            bot.position_mgr.positions.pop(symbol, None)
            continue

        broker_qty = abs(int(float(broker_pos.get("qty", 0))))
        if broker_qty <= 0:
            logger.warning(f"EXIT SKIP {symbol}: broker qty=0 — removing local")
            bot.position_mgr.positions.pop(symbol, None)
            continue

        qty = min(position.quantity, broker_qty)
        if qty != position.quantity:
            logger.warning(
                f"EXIT {symbol}: local qty={position.quantity} but broker qty={broker_qty} — selling {qty}"
            )
            position.quantity = qty

        # Market sell
        sell_resp = bot.position_mgr._submit_sell_order(symbol, qty)
        if not sell_resp:
            # Limit fallback
            last_price = bot.position_mgr._get_last_price(symbol)
            if last_price and last_price > 0:
                limit_price = bot.position_mgr.round_limit_price(last_price * 0.97)
                logger.warning(f"Market sell failed for {symbol}, trying limit @ {limit_price}")
                sell_resp = bot.position_mgr._submit_sell_order(symbol, qty, "limit", limit_price)

        if sell_resp and sell_resp.get("id"):
            order_id = sell_resp["id"]
            submitted_orders.append((order_id, symbol, qty))
        else:
            logger.error(f"Failed to submit sell for {symbol} x{qty}")

    # Pass 2: monitor fills
    for order_id, symbol, qty in submitted_orders:
        position = bot.position_mgr.positions.get(symbol)
        if not position:
            continue

        fill = bot.position_mgr.get_order_fill(order_id, max_wait=30)
        if fill and int(fill.get("filled_qty", 0)) > 0:
            filled_qty = int(fill["filled_qty"])
            exit_price = fill["filled_avg_price"]

            # PDT guard
            if filled_qty > 0:
                bot.sold_today.add(symbol)

            remaining = position.quantity - filled_qty
            if remaining > 0:
                position.quantity = remaining
                # Immediate resubmit for residual (same logic as old _exit_position)
                slice_residual = qty - filled_qty
                fill_status = fill.get("status", "unknown")
                if slice_residual > 0 and fill_status != "filled":
                    logger.warning(
                        f"EXIT PARTIAL {symbol}: filled {filled_qty}/{qty} — resubmitting {slice_residual}"
                    )
                    resub_resp = bot.position_mgr._submit_sell_order(symbol, slice_residual)
                    if resub_resp and resub_resp.get("id"):
                        resub_id = resub_resp["id"]
                        resub_fill = bot.position_mgr.get_order_fill(resub_id, max_wait=15)
                        if resub_fill:
                            resub_qty = int(resub_fill.get("filled_qty", 0))
                            remaining -= resub_qty
                            position.quantity = remaining
                            if resub_qty > 0:
                                bot.sold_today.add(symbol)
                            logger.info(
                                f"EXIT RESUBMIT {symbol}: filled additional {resub_qty}, now {remaining} remaining"
                            )
                        else:
                            logger.warning(f"EXIT RESUBMIT {symbol}: no fill on resubmit order")
                    else:
                        logger.error(f"EXIT RESUBMIT {symbol}: resubmit failed")
            else:
                bot.position_mgr.positions.pop(symbol, None)
                logger.info(f"EXIT FILLED {symbol}: qty={filled_qty}, price={exit_price:.4f}")
        else:
            logger.warning(f"EXIT NO FILL {symbol}: keeping for retry")

    # Reconcile local state with broker
    actions = bot.position_mgr.reconcile_local_positions_from_broker()
    if actions:
        logger.info(f"EXIT {sleeve}: post-exit reconciliation adjustments: {actions}")

    # Count remaining positions (include UNKNOWN in GDP count for safety)
    if sleeve == "GDP":
        remaining = sum(
            1 for p in bot.position_mgr.positions.values()
            if getattr(p, "sleeve", "MR") in ("GDP", "UNKNOWN")
        )
    else:
        remaining = sum(
            1 for p in bot.position_mgr.positions.values()
            if getattr(p, "sleeve", "MR") == sleeve
        )
    logger.info(f"EXIT {sleeve}: done — {remaining} positions remaining")


def exit_single_position(bot, symbol: str, reason: str) -> None:
    """Exit a single position — delegates to position_mgr._exit_position().

    That method handles: broker position check -> market sell -> limit
    fallback -> partial fill resubmit -> local state cleanup.
    We check whether the symbol is fully gone afterwards and persist state.
    """
    if symbol not in bot.position_mgr.positions:
        return

    result = bot.position_mgr._exit_position(symbol, reason)

    # PDT guard: only mark as sold if shares actually changed hands
    if result.get("filled_qty", 0) > 0:
        bot.sold_today.add(symbol)

    still_held = symbol in bot.position_mgr.positions
    if still_held:
        remaining = bot.position_mgr.positions[symbol].quantity
        failsafe_t = (
            config.RED_OPEN_TRAIL_FAILSAFE_TIME
            if getattr(config, "ENABLE_RED_OPEN_TRAIL_EXIT", False)
            else config.V2_FAILSAFE_TIME
        )
        logger.warning(
            f"EXIT INCOMPLETE {symbol}: {remaining} shares still held — failsafe will catch at {failsafe_t}"
        )
    bot._save_state()


# ────────────────────────────────────────────────────────────────────
# Open exit plan + batched market sells
# ────────────────────────────────────────────────────────────────────

def build_open_exit_plan_from_broker(bot, reason: str = "broker snapshot") -> List[Dict[str, Any]]:
    """Freeze the remaining broker positions into a simple 09:30 market-sell plan.

    This is intentionally called before the open, after canceling premarket
    limit orders. The 09:30 path should not fetch prices or make
    green/red decisions; it should only submit these prepared market sells.
    """
    broker_positions = bot.position_mgr.get_broker_positions()
    if broker_positions is None:
        logger.error("OPEN_EXIT_PLAN: broker position read failed during %s", reason)
        bot.open_exit_plan = []
        return bot.open_exit_plan

    plan: List[Dict[str, Any]] = []
    for broker_pos in broker_positions:
        symbol = str(broker_pos.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        try:
            qty = int(abs(float(broker_pos.get("qty", 0))))
        except (TypeError, ValueError):
            logger.warning("OPEN_EXIT_PLAN: bad qty for %s: %s", symbol, broker_pos.get("qty"))
            continue
        if qty <= 0:
            continue

        plan.append({
            "symbol": symbol,
            "qty": qty,
            "avg_entry_price": broker_pos.get("avg_entry_price"),
            "source": reason,
        })

    bot.open_exit_plan = plan
    logger.warning(
        "OPEN_EXIT_PLAN_READY reason=%s count=%d symbols=%s",
        reason,
        len(plan),
        [p["symbol"] for p in plan],
    )
    return plan


def submit_open_exit_market_sells(bot) -> None:
    """09:30 fast liquidation: submit all market sells first, then reconcile.

    No snapshots. No green/red branch. No trailing stops. No fill-wait inside
    the submit loop. This keeps the 09:30 minute focused purely on order
    submission.
    """
    if not bot.open_exit_plan:
        logger.warning("OPEN_EXIT: no frozen 09:25 plan found; building one from broker now")
        build_open_exit_plan_from_broker(bot, reason="09:30 fallback broker snapshot")

    if not bot.open_exit_plan:
        logger.warning("OPEN_EXIT: no broker positions to submit")
        bot.position_mgr.reconcile_local_positions_from_broker()
        return

    logger.warning(
        "OPEN_EXIT_BATCH_START count=%d symbols=%s",
        len(bot.open_exit_plan),
        [p["symbol"] for p in bot.open_exit_plan],
    )

    submitted_orders: List[Tuple[str, str, int]] = []

    # Phase 1: submit every market sell as quickly as possible.
    for item in list(bot.open_exit_plan):
        symbol = item.get("symbol")
        qty = int(item.get("qty", 0) or 0)
        if not symbol or qty <= 0:
            continue

        submit_start = datetime.now(_ET)
        t0 = time.perf_counter()
        try:
            resp = bot.position_mgr._submit_sell_order(
                symbol,
                qty,
                order_type="market",
                time_in_force="day",
                extended_hours=False,
            )
        except TypeError:
            # Backward compatibility with older PositionManager signatures.
            resp = bot.position_mgr._submit_sell_order(symbol, qty)
        except Exception:
            logger.exception("OPEN_EXIT_SUBMIT_FAILED symbol=%s qty=%s", symbol, qty)
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        order_id = resp.get("id") if resp else None
        if order_id:
            submitted_orders.append((order_id, symbol, qty))
            bot.sold_today.add(symbol)
            logger.warning(
                "OPEN_EXIT_SUBMITTED symbol=%s qty=%s order_id=%s start=%s elapsed_ms=%.1f type=market tif=day ext=false",
                symbol,
                qty,
                order_id,
                submit_start.isoformat(),
                elapsed_ms,
            )
        else:
            logger.error("OPEN_EXIT_SUBMIT_NO_ORDER_ID symbol=%s qty=%s resp=%s", symbol, qty, resp)

    logger.warning(
        "OPEN_EXIT_BATCH_ALL_SUBMITTED submitted=%d planned=%d",
        len(submitted_orders),
        len(bot.open_exit_plan),
    )

    # Phase 2: do not block the opening submissions. Reconcile once after all
    # orders are out; the normal failsafe can catch anything still held.
    actions = bot.position_mgr.reconcile_local_positions_from_broker()
    if actions:
        logger.info("OPEN_EXIT: post-submit reconciliation adjustments: %s", actions)

    remaining = bot.position_mgr.broker_position_count()
    if remaining == 0:
        logger.warning("OPEN_EXIT: broker confirmed flat after batch submit")
        bot.position_mgr.positions.clear()
        bot.open_exit_plan = []
    elif remaining > 0:
        logger.warning("OPEN_EXIT: broker still shows %d positions after batch submit; failsafe will retry", remaining)
    else:
        logger.error("OPEN_EXIT: broker position count unavailable after batch submit")


def run_broker_native_rescue(bot) -> None:
    """Broker-native rescue pass at 09:31 for any remaining positions.

    Fetches broker positions directly and market-sells whatever remains.
    Submits all orders first, then reconciles with a small sleep for Alpaca to update state.
    Only marks exits complete after broker confirms flat.
    """
    broker_positions = bot.position_mgr.get_broker_positions()
    if broker_positions is None:
        logger.error("09:31 rescue: broker position read failed; skipping")
        return

    if not broker_positions:
        logger.info("09:31 rescue: broker confirmed flat")
        return

    logger.warning(f"09:31 rescue: broker has {len(broker_positions)} positions, selling all")

    submitted: List[Tuple[str, int, str]] = []

    for broker_pos in broker_positions:
        symbol = str(broker_pos.get("symbol", "")).upper()
        try:
            qty = abs(int(float(broker_pos.get("qty", 0))))
        except (TypeError, ValueError):
            logger.warning(f"09:31 rescue: bad qty for {symbol}: {broker_pos.get('qty')}")
            continue

        if qty <= 0:
            continue

        logger.info(f"09:31 rescue: selling {symbol} x{qty}")
        resp = bot.position_mgr._submit_sell_order(
            symbol=symbol,
            qty=qty,
            order_type="market",
            time_in_force="day",
            extended_hours=False,
        )

        if resp and resp.get("id"):
            submitted.append((symbol, qty, resp["id"]))
            logger.info(f"09:31 rescue: submitted market sell for {symbol} x{qty}, order_id={resp['id']}")
        else:
            logger.error(f"09:31 rescue: failed to submit market sell for {symbol} x{qty}")

    logger.warning(f"09:31 rescue: submitted={len(submitted)}")

    # Small sleep to allow Alpaca to update position state
    time.sleep(3)

    # Reconcile after rescue
    bot.position_mgr.reconcile_local_positions_from_broker()
    remaining = bot.position_mgr.broker_position_count()
    if remaining == 0:
        logger.info("09:31 rescue: broker confirmed flat after rescue")
    else:
        logger.warning(f"09:31 rescue: broker still has {remaining} positions after rescue")


# ────────────────────────────────────────────────────────────────────
# Red-open trailing-stop decision + failsafe flatten
# ────────────────────────────────────────────────────────────────────

def position_reference_price(bot, broker_pos: dict, symbol: str) -> Optional[float]:
    """Best available 09:30 decision price for red-open trailing logic.

    Prefer a fresh snapshot/last price at decision time — the broker
    position endpoint's current_price can lag around the open. Fall back
    to broker position fields only if the snapshot call fails.
    """
    try:
        px = bot.position_mgr._get_last_price(symbol)
        if px and px > 0:
            return float(px)
    except Exception:
        logger.warning(f"RED TRAIL {symbol}: failed to fetch last price", exc_info=True)

    for key in ("current_price", "lastday_price"):
        raw = broker_pos.get(key) if broker_pos else None
        try:
            px = float(raw)
            if px > 0:
                return px
        except (TypeError, ValueError):
            pass

    return None


def submit_red_open_trail_or_sell_green(bot) -> None:
    """09:30 exit decision for the red-open trailing-stop paper test.

    Batch behavior:
    - Trailing-stop orders are submitted without waiting.
    - Green/flat market exits are all submitted first, then monitored.
    - If a position is green/flat versus its stored entry price, sell it at market.
    - If a position is red versus its stored entry price, submit an Alpaca
      trailing-stop sell order using config.RED_OPEN_TRAIL_PCT.
    - Any bad/missing price or qty data falls back to an immediate market sell.
    - The failsafe at RED_OPEN_TRAIL_FAILSAFE_TIME cancels any remaining
      trailing orders and force-flattens.
    """
    if bot.red_trail_exit_submitted:
        logger.info("RED TRAIL: 09:30 exit decision already submitted; skipping duplicate call")
        return

    broker_positions = bot.position_mgr.get_broker_positions()
    if broker_positions is None:
        logger.error("RED TRAIL: cannot read broker positions; falling back to local market exits")
        for symbol in list(bot.position_mgr.positions.keys()):
            exit_single_position(bot, symbol, "red-trail broker read failed fallback")
        bot.red_trail_exit_submitted = True
        bot._save_state()
        return

    broker_by_symbol = {
        str(p.get("symbol", "")).upper(): p
        for p in broker_positions
        if p.get("symbol")
    }

    # Reconcile first so UNKNOWN broker-only positions are included in this decision.
    bot.position_mgr.reconcile_local_positions_from_broker()

    submitted_trails = 0
    market_exits = 0
    fallbacks = 0
    market_orders: Dict[str, Dict[str, Any]] = {}

    # Pass 1: submit trailing stops and collect market exit symbols
    for symbol, pos in list(bot.position_mgr.positions.items()):
        sym = symbol.upper()
        broker_pos = broker_by_symbol.get(sym)

        if not broker_pos:
            logger.warning(f"RED TRAIL {symbol}: broker no longer holds symbol; removing local position")
            bot.position_mgr.positions.pop(symbol, None)
            continue

        try:
            broker_qty = int(abs(float(broker_pos.get("qty", pos.quantity))))
        except (TypeError, ValueError):
            broker_qty = int(pos.quantity or 0)

        entry_price = float(getattr(pos, "entry_price", 0.0) or broker_pos.get("avg_entry_price", 0.0) or 0.0)
        decision_price = position_reference_price(bot, broker_pos, symbol)

        if broker_qty <= 0:
            logger.warning(f"RED TRAIL {symbol}: broker qty <= 0; removing local position")
            bot.position_mgr.positions.pop(symbol, None)
            continue

        if entry_price <= 0 or not decision_price or decision_price <= 0:
            logger.warning(
                f"RED TRAIL {symbol}: bad decision data "
                f"entry={entry_price}, price={decision_price}; submitting market fallback"
            )
            resp = bot.position_mgr._submit_sell_order(symbol, broker_qty)
            if resp and resp.get("id"):
                market_orders[symbol] = {"order_id": resp["id"], "qty": broker_qty}
                market_exits += 1
            else:
                fallbacks += 1
            continue

        # Optional tiny buffer. Default 0.0 means any price below entry is red.
        red_trigger_price = entry_price * (1.0 - config.RED_OPEN_TRAIL_PRICE_BUFFER_PCT)

        if decision_price < red_trigger_price:
            resp = bot.position_mgr.submit_trailing_stop_sell_order(
                symbol=symbol,
                qty=broker_qty,
                trail_percent=config.RED_OPEN_TRAIL_PCT,
            )
            if resp and resp.get("id"):
                order_id = resp["id"]
                bot.red_trail_order_ids[symbol] = order_id
                bot.red_trail_symbols.add(symbol)
                bot.sold_today.add(symbol)
                submitted_trails += 1
                logger.info(
                    f"RED TRAIL {symbol}: price={decision_price:.4f} < entry={entry_price:.4f}; "
                    f"submitted {config.RED_OPEN_TRAIL_PCT:.2f}% trailing-stop sell "
                    f"qty={broker_qty} order_id={order_id}"
                )
            else:
                logger.error(f"RED TRAIL {symbol}: trailing-stop submit failed; submitting market fallback")
                resp = bot.position_mgr._submit_sell_order(symbol, broker_qty)
                if resp and resp.get("id"):
                    market_orders[symbol] = {"order_id": resp["id"], "qty": broker_qty}
                    market_exits += 1
                else:
                    fallbacks += 1
        else:
            logger.info(
                f"GREEN/FLAT EXIT {symbol}: price={decision_price:.4f} >= entry={entry_price:.4f}; "
                f"submitting market sell"
            )
            resp = bot.position_mgr._submit_sell_order(symbol, broker_qty)
            if resp and resp.get("id"):
                market_orders[symbol] = {"order_id": resp["id"], "qty": broker_qty}
                market_exits += 1
            else:
                fallbacks += 1

    # Pass 2: monitor green/flat market exits only after all orders are in
    for symbol, meta in market_orders.items():
        pos = bot.position_mgr.positions.get(symbol)
        if not pos:
            continue
        fill = bot.position_mgr.get_order_fill(meta["order_id"], max_wait=20)
        if fill and int(fill.get("filled_qty", 0)) > 0:
            filled_qty = int(fill["filled_qty"])
            bot.sold_today.add(symbol)
            remaining = int(pos.quantity) - filled_qty
            if remaining > 0:
                pos.quantity = remaining
                logger.warning(f"GREEN/FLAT EXIT {symbol}: partial fill {filled_qty}; {remaining} remaining")
            else:
                logger.info(f"GREEN/FLAT EXIT {symbol}: filled {filled_qty}; removing local position")
                bot.position_mgr.positions.pop(symbol, None)
        else:
            logger.warning(f"GREEN/FLAT EXIT {symbol}: no confirmed fill yet — failsafe will retry")

    bot.position_mgr.reconcile_local_positions_from_broker()
    bot.red_trail_exit_submitted = True
    logger.warning(
        f"RED TRAIL DECISION COMPLETE: trailing_orders={submitted_trails}, "
        f"market_exits={market_exits}, fallbacks={fallbacks}, "
        f"open_trail_symbols={list(bot.red_trail_order_ids.keys())}"
    )
    bot._save_state()


def run_failsafe_flatten(bot, label: str) -> None:
    """Broker-based catch-all flatten with multi-layer retry."""
    logger.warning(f"{label}: starting broker-based failsafe flatten")
    if bot.red_trail_order_ids:
        logger.warning(
            f"{label}: canceling open trailing-stop orders before force flatten: "
            f"{list(bot.red_trail_order_ids.keys())}"
        )
        bot.position_mgr.cancel_all_open_orders()
        bot.sold_today.update(bot.red_trail_order_ids.keys())

    summary = bot.position_mgr.force_flatten_broker_positions(label)

    logger.warning(
        f"{label}: failsafe flatten complete | "
        f"positions_seen={summary['positions_seen']} | "
        f"closes_submitted={summary['closes_submitted']} | "
        f"fills_confirmed={summary['fills_confirmed']} | "
        f"errors={len(summary['errors'])}"
    )

    manual = summary.get("manual_required", [])
    if manual:
        for item in manual:
            logger.critical(f"{label}: {item}")

    remaining = bot.position_mgr.broker_position_count()
    if remaining == 0:
        bot.position_mgr.positions.clear()
        bot.red_trail_order_ids.clear()
        bot.red_trail_symbols.clear()
        logger.warning(f"{label}: broker confirmed flat — local state cleared")
    elif remaining < 0:
        logger.error(f"{label}: broker API unreachable after failsafe — cannot confirm flat")
    else:
        logger.error(f"{label}: broker still shows {remaining} open positions after failsafe")

    bot._save_state()
