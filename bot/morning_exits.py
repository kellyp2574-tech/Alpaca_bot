"""Morning exit pipeline — extracted from integrated_main.py.

Owns the 09:25 → ~09:45 exit sequence: open-exit plan construction at 09:25,
batched market-sell submission at 09:30, broker-native rescue at 09:31, and
the 09:45 failsafe flatten.

Every function takes the orchestrator instance (`bot`) as its first argument
and mutates its state directly. State ownership stays with
`CombinedOvernightReboundBot`; this module only houses the logic.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from bot import config

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


# ────────────────────────────────────────────────────────────────────
# Sleeve / single-position exits  (REMOVED)
# ────────────────────────────────────────────────────────────────────
# `exit_sleeve_positions`, `exit_single_position`, `position_reference_price`
# and `submit_red_open_trail_or_sell_green` were the legacy GDP/MR-split
# exit branches and the red-open trailing-stop paper experiment. Both are
# disabled in production: the live path is the fast batched market-sell at
# 09:30 (`submit_open_exit_market_sells`) followed by the 09:45 failsafe
# (`run_failsafe_flatten`).



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

        # Duplicate-sell guard: if a sell order already exists (e.g. bot
        # restarted after submitting but before broker confirmed flat),
        # skip re-submitting to avoid doubling the sell quantity.
        try:
            open_orders = bot.position_mgr.list_open_orders(symbols=[symbol])
            if open_orders and any(
                str(o.get("side", "")).lower() == "sell" for o in open_orders
            ):
                logger.warning(
                    "OPEN_EXIT: sell already open for %s — skipping duplicate submit", symbol
                )
                bot.sold_today.add(symbol)
                continue
        except Exception:
            logger.warning("OPEN_EXIT: could not check open orders for %s; submitting anyway", symbol, exc_info=True)

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
        bot.overnight_etf_position = None
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

        # Issue 6: Check for existing open sell orders to avoid duplicate submissions
        open_orders = bot.position_mgr.list_open_orders(symbols=[symbol])
        if open_orders and any(order.get("side") == "sell" for order in open_orders):
            logger.warning(f"09:31 rescue: sell already open for {symbol}; not submitting duplicate")
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

def run_failsafe_flatten(bot, label: str) -> None:
    """Broker-based catch-all flatten with multi-layer retry."""
    logger.warning(f"{label}: starting broker-based failsafe flatten")

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
        bot.overnight_etf_position = None
        logger.warning(f"{label}: broker confirmed flat — local state cleared")
    elif remaining < 0:
        logger.error(f"{label}: broker API unreachable after failsafe — cannot confirm flat")
    else:
        logger.error(f"{label}: broker still shows {remaining} open positions after failsafe")

    bot._save_state()
